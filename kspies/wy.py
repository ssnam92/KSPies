"""
Wu and Yang
===========

**Summary** This is a program for preforming Kohn-Sham inversion with algorithm introduced by Wu and Yang [WuYang2003]_.

Written for Python 3.7.4 Lots of other text here, like details about how this uses the original Wu Yang idea [WuYang2003]_, and how the other ref is used for  Regularization [HBY2007]_ . Any useful info about the algorithm should also be given.

.. moduleauthor::
    Seungsoo Nam <skaclitz@yonsei.ac.kr> <http://tccl.yonsei.ac.kr/mediawiki/index.php/Main_Page> ORCID: `000-0001-9948-6140 <https://orcid.org/0000-0001-9948-6140>`_
    Ryan J. McCarty <rmccarty@uci.edu> <http://ryanjmccarty.com> ORCID: `0000-0002-2417-8917 <http://https://orcid.org/0000-0002-2417-8917>`_

:References:

    .. [WuYang2003] Qin Wu and Weitao Yang. A direct optimization method for calculating density functionals and exchange–correlation potentials from electron densities (2003)
        <https://doi.org/10.1063/1.1535422> Journal of Chemical Physics, 118(6).

    .. [HBY2007] Tim Heaton-Burgess, Felipe A. Bulat, and Weitao Yang. Optimized Effective Potentials in Finite Basis Sets (2007)
        <https://doi.org/10.1103/PhysRevLett.98.256401> Physical review letters, 98(25).

    .. [GNGK2020] Rachel Garrick, Amir Natan, Tim Gould, and Leeor Kronik. Exact Generalized Kohn-Sham Theory for Hybrid Functionals (2020)
        <https://doi.org/10.1103/PhysRevX.10.021040> Physical Review X, 10(2).

.. topic:: Funding

    This research was made possible by funding from the National Research Foundation of Korea (NRF-2020R1A2C2007468 and NRF-2020R1A4A1017737).

.. topic:: Internal Log

    **2020-06-02** RJM added in docstring templates

    **2020-06-15** SN added some comments and fix bug(RWY and UWY gave different results with regularization)

    **2020-08-21** SN corrected typos, minor changes in attribute names, etc.

"""

import time
import numpy as np
from scipy.optimize import minimize
from pyscf import scf, dft
try:
    import kspies_fort
    kf_imported=True
except:
    kf_imported=False

def numint_3c2b(mol, pbas, level=5):
    """Summary: Three-center overlap integral with different atomic- and potential- basis sets with numerical integration

    Args:
        mol : an instance of :class:`Mole`
        pbas (dict or str) : potential basis set for WY
        level (int 0~9) : PySCF preset mesh grid used for numerical integration. Default is 5

    Returns:
        (ndarray): **Sijt** three-center overlap integral with shape ((n1, n1, n2))
        where n1 is the length of atomic orbital basis set defined in mol,
        and n2 is the length of potential basis set

    """
    mol2 = mol.copy()
    mol2.basis = pbas
    mol2.build()
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    coords = grids.coords
    weights = grids.weights
    ao1 = dft.numint.eval_ao(mol, coords, deriv=0)
    ao2 = dft.numint.eval_ao(mol2, coords, deriv=0)
    t0 = time.time()
    if kf_imported: #Utilize faster numerical integration
        Sijt = kspies_fort.ovlp_aab(weights, ao1, ao2, mol.nao_nr(), mol2.nao_nr(), len(weights))
    else: #kspies_fort not imported, us numpy
        Sijt = np.einsum('ri, rj, rk, r->ijk', ao1, ao1, ao2, weights, optimize = 'greedy')
    n1 = mol.nao_nr()
    n2 = mol2.nao_nr()
    t = (time.time()-t0)/60.
    print("Three-center overlap integral by numerical integration")
    print("n1  : %4i"%n1)
    print("n2  : %4i"%n2)
    print("time: %.1f min"%t)
    return Sijt

def time_profile(mw):
    """Summary: Print to terminal a collection of time data in seconds

    Terminal prints:
    * initialize : time for initialization
    * total_opt  : total elasped time
    * solve_eig  : time to construct fock matrix and diagonalize it
    * Ws_eval    : time to evaluate objective function Ws
    * Gd_eval    : time to evaluate gradient of objective function

    """
    print("*********Time Profile*********")
    print("initialize : %15.3f"%mw.t_init)
    print("total_opt  : %15.3f"%mw.t_opt)
    print("solve_eig  : %15.3f"%mw.t_eig)
    print("Ws_eval    : %15.3f"%mw.t_ws)
    print("Gd_eval    : %15.3f"%mw.t_gd)
    print("Hs_eval    : %15.3f"%mw.t_hs)

def run(mw):
    """Summary: Minimize an objective function -Ws

    Args:
        mw : RWY or UWY object

    """
    mw.initialize()
    t = time.time()
    if mw.method.lower()=='cg' or mw.method.lower()=='bfgs':
        mw.res = minimize(mw.eval_Ws, x0=mw.b, method=mw.method, jac=mw.eval_Gd,
                          tol=mw.tol, options={'disp': False})
    else:
        mw.res = minimize(mw.eval_Ws, x0=mw.b, method=mw.method, jac=mw.eval_Gd, hess=mw.eval_Hs,
                          tol=mw.tol, options={'disp': False})
    mw.b = mw.res.x
    mw.converged = mw.res.success
    mw.t_opt += time.time()-t

def wyscf(mw, ddmtol=1e-6):
    """Summary: Run WY until its self-consistent. This is required for hybrid guiding potentials

    .. note::
        This function performs very simplied generalization of WY for non-local guiding potential, mimicking [GNGK2020]_.
        There is no theoretical proof that the result of this function can be trusted.
        The optimization of WY objective functional is only theoretically proven for local multiplicative potentials, not non-local potential.

    Args:
        mw : RWY or UWY object
        ddmtol (float, optional) : Convergence criteria of density matrix.

    """
    mw.totnit = 0
    mw.run()
    mw.totnit += mw.res.nit
    mw.scfcycle = 0
    for i in range(40):
        mw.scfcycle = i+1
        dm_old = mw.dm
        mw.dm_aux = dm_old
        mw.run()
        mw.totnit += mw.res.nit
        mw.ddm = np.max(np.abs(np.array(dm_old)-np.array(mw.dm)))
        if mw.ddm < ddmtol:
            break

def info(mw):
    """Summary: Print summary of minimization

    Args:
        mw : RWY or UWY object

    """
    grd = mw.res.jac
    maxgrd = np.max(abs(grd))
    if mw.converged == False:
        print(" *****Optimization Failed*****")
        print("      after %i iterations "%mw.res.nit)
    else:
        print("****Optimization Completed****")
        if hasattr(mw, 'scfcycle'):
            print("      after %i scf cycles "%mw.scfcycle)
            print("      and %i iterations "%mw.totnit)
            print("max(ddm)   : %15.8f"%mw.ddm)
        else:
            print("      after %i iterations "%mw.res.nit)
    print("func_value : %15.8f"%mw.res.fun)
    print("max_grad   : %15.8f"%maxgrd)

def basic(mw, mol, pbas, Sijt):
    """Summary: Common basic initialization function for RWY and UWY objects

    Args:
        mw : RWY or UWY object
        mol : an instance of :class:`Mole`
        pbas (dict or str) : to define potential basis set for WY
        Sijt (ndarray) : three-center overlap integral with shape ((n1, n1, n2))
          where n1 is the length of atomic orbital basis set defined in mol,
          while n2 is the length of potential basis set

    """

    mw.mol = mol
    mw.guide = 'faxc'
    mw.reg = 0.
    mw.tol = 1e-6
    mw.method = 'trust-exact'
    mw.verbose = mw.mol.verbose
    mw.stdout = mw.mol.stdout

    is_model1 = len(mw.mol._atm) == 0
    is_model2 = len(mw.mol._bas) == 0
    mw.model = is_model1 and is_model2
    if mw.model: #Check if defined mol is molecule or model
        if Sijt is None:
            raise AssertionError("Three-center overlap integeral should be given for model system")
        if kf_imported:
            mw.Sijt = np.array(Sijt, order='F')
        else:
            mw.Sijt = np.array(Sijt, order='C')
        return

    mw.S = mw.mol.intor_symmetric('int1e_ovlp')
    mw.T = mw.mol.intor_symmetric('int1e_kin')
    mw.V = mw.mol.intor_symmetric('int1e_nuc')

    def mol2build(pbas):
        mw.mol2 = mol.copy()
        mw.mol2.basis = pbas
        mw.mol2.build()

    if pbas is None and Sijt is not None:
        raise AssertionError("potential basis should be given for given Sijt")

    if pbas is None and Sijt is None:
        mol2build(mol.basis)
        mw.Sijt = mol.intor('int3c1e')
    elif pbas is not None and Sijt is None:
        mol2build(pbas)
        mw.Sijt = numint_3c2b(mol, pbas)
    else:
        mol2build(pbas)
        is_same_pbas = mw.mol2.nao_nr() == len(Sijt[0, 0, :])
        is_same_bas = mw.mol.nao_nr() == len(Sijt[:, 0, 0])
        if is_same_pbas and is_same_bas:
            mw.Sijt = Sijt
        else:
            raise AssertionError("dimension of given basis are not consistent with Sijt")
    mw.Tp = mw.mol2.intor_symmetric('int1e_kin')

class RWY:
    """Summary: Perform WY calculation in restricted scheme

    .. _restrictedwy:

    Attributes:
        mol (object) : an instance of :class:`Mole`
        dm_tar (ndarray) : Density matrix of target density in atomic orbital basis representation
        pbas (dict or str) : Potential basis set for WY. If not given, same with atomic orbital basis
        Sijt (ndarray) : Three-center overlap integral. If not given, calculated analytically as overlap of atomic orbital basis
        dm_aux (ndarray) : Auxilary density matrix to construct density-dependent part of guiding potential. Default is dm_tar
        guide (str) : Guiding potential. Can be set as

            |  None : no guiding potential except external potential
            |  'faxc' : Exchange-correlation part of Fermi-Amaldi potential
            |  xc   : ks.xc attribute in pyscf DFT

        reg (float) : strength of regularization. Default is 0
        tol (float) : tolarance of optimization (maximum gradient element). Default is 1e-6
        method (str) : optimization algorithm used in SciPy. Can be (case insensitive) 'cg', 'bfgs', 'newton-cg',
                       'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'. Defult is 'trust-exact'

    :ivar:
        * **converged** (bool) – optimization converged or not
        * **mo_energy** (ndarray) –  molecular orbital energies
        * **mo_occ** (ndarray) –  molecular orbital occupation numbers
        * **mo_coeff** (ndarray) – molecular orbital coefficients
        * **dm** (ndarray) – density matrix in atomic orbital basis representation
        * **b** (ndarray) – optimized linear combination coefficients

    """
    def __init__(self, mol, dm_tar, pbas=None, Sijt=None, dm_aux=None):
        t = time.time()
        basic(self, mol, pbas, Sijt)

        self.nbas = len(self.Sijt[:, 0, 0])
        self.npot = len(self.Sijt[0, 0, :])
        self.b = np.zeros(self.npot)
        self.internal_b = np.ones_like(self.b)
        self.dm_aux = dm_aux
        self.dm_tar = dm_tar
        self.t_eig = self.t_ws = self.t_gd = self.t_hs = self.t_opt = 0.
        self.t_init = time.time()-t

    get_occ = scf.hf.get_occ
    make_rdm1 = scf.hf.SCF.make_rdm1

    run = run
    time_profile = time_profile
    info = info
    wyscf = wyscf

    def initialize(self):
        """Summary: Construct a fixed part of fock matrix F0

        .. math::

            F_{0} = T + V_{ext} + V_{h} + V_{g}

        """
        self.get_ovlp = lambda *args: self.S
        self.internal_b = np.ones_like(self.b)

        if self.dm_aux is None:
            self.dm_aux = self.dm_tar

        if self.guide is None:
            self.V0 = np.zeros_like(self.dm_tar)
        elif self.guide.lower() == 'faxc':
            N = self.mol.nelectron
            J_tar = scf.hf.get_jk(self.mol, self.dm_aux)[0]
            self.V0 = ((N-1.)/N)*(J_tar)
        else:
            mydft = dft.RKS(self.mol)
            mydft.xc = self.guide
            self.V0 = mydft.get_veff(self.mol, dm=self.dm_aux)
        self.F0 = self.T+self.V+self.V0

    def solve(self, b):
        """Summary: Under a given b vector, construct a fock matrix and diagonalize it

        F = F0 + V(b)

        FC = SCE

        resulting mo_coeff(C), mo_energy(E), and density matrix are stored as instance attributes
        """
        if not np.allclose(b, self.internal_b, rtol=1e-12, atol=1e-12):
            t = time.time()
            self.internal_b = b.copy()
            self.fock = self.F0+np.einsum('t,ijt->ij', b, self.Sijt)
            self.mo_energy, self.mo_coeff = scf.hf.eig(self.fock, self.S)
            self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
            self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
            self.t_eig += time.time()-t

            t = time.time()
            if kf_imported:
                self.grad = kspies_fort.einsum_ij_ijt_2t((self.dm-self.dm_tar), self.Sijt, self.nbas, self.npot)
            else:
                self.grad = np.einsum('ij,ijt->t', (self.dm-self.dm_tar), self.Sijt)
            self.t_gd += time.time()-t

    def eval_Ws(self, b):
        """Summary: Calculate objective function Ws under given b vector

        When mw.reg > 0, regularization is added
        """
        self.solve(b)
        t = time.time()

        Ws = np.einsum('ij,ji', self.T, self.dm)
        Ws += np.einsum('ij,ji', (self.V+self.V0), (self.dm-self.dm_tar))
        Ws += np.einsum('t,t', b, self.grad)
        self.Ws = Ws

        penelty = 0.
        if self.reg > 0:
            penelty = self.reg*self.Dvb(b)

        self.t_ws += time.time()-t
        return -(Ws-penelty)

    def eval_Gd(self, b):
        """Summary: -d(Ws)/b_t
        """
        self.solve(b)
        t = time.time()
        self.Gd = self.grad

        if self.reg > 0:
            self.Gd -= 4*self.reg*np.einsum('st,t->s', self.Tp, b)
        self.t_gd += time.time()-t
        return -self.Gd

    def eval_Hs(self, b):
        """Summary: -d^2(Ws)/(b_t)(b_u)
        """
        self.solve(b)
        t = time.time()

        nocc = self.mol.nelectron//2
        eia = self.mo_energy[:nocc, None] - self.mo_energy[None, nocc:]

        if kf_imported:
            self.Hs=kspies_fort.wy_hess(self.Sijt,self.mo_coeff,eia,nocc,self.nbas-nocc,self.npot)
        else:
            Siat = np.einsum('mi,va,mvt->iat',
                             self.mo_coeff[:,:nocc],self.mo_coeff[:,nocc:],self.Sijt)
            self.Hs= 4*np.einsum('iau,iat,ia->ut',Siat,Siat,eia**-1)

        if self.reg > 0:
            self.Hs -= 4*self.reg*self.Tp
        self.t_hs += time.time()-t
        return -self.Hs

    def Dvb(self, b=None):
        """Summary: Calcuate the norm of the correction potential derivative

        .. math::

            \\| \\nabla v_C ( \\textbf{r} ) \\|^2 = \\int v_C ( \\textbf{r} ) \\nabla^2 v_C ( \\textbf{r} ) d \\textbf{r}

        """
        if b is None:
            b = self.b
        Dvb = 2*np.einsum('s,st,t', b, self.Tp, b)
        return Dvb

class UWY:
    """Summary: Perform WY calculation in unrestricted scheme

    .. _unrestrictedwy:

    Attributes:
        mol (object) : an instance of :class:`Mole`
        dm_tar (ndarray) : Density matrix of target density in atomic orbital basis representation
        pbas (dict or str) : Potential basis set for WY. If not given, same with atomic orbital basis
        Sijt (ndarray) : Three-center overlap integral. If not given, calculated analytically as overlap of atomic orbital basis
        dm_aux (ndarray) : Auxilary density matrix to construct density-dependent part of guiding potential. Default is dm_tar
        guide (str) : Guiding potential. Can be set as

            |  None : no guiding potential except external potential
            |  'faxc' : Exchange-correlation part of Fermi-Amaldi potential
            |  xc   : ks.xc attribute in pyscf DFT

        reg (float) : strength of regularization. Default is 0
        tol (float) : tolarance of optimization (maximum gradient element). Default is 1e-6
        method (str) : optimization algorithm used in SciPy. Can be (case insensitive) 'cg', 'bfgs', 'newton-cg',
                       'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'. Defult is 'trust-exact'

    :ivar:
        * **converged** (bool) – optimization converged or not
        * **mo_energy** (ndarray) –  molecular orbital energies
        * **mo_occ** (ndarray) –  molecular orbital occupation numbers
        * **mo_coeff** (ndarray) – molecular orbital coefficients
        * **dm** (ndarray) – density matrix in atomic orbital basis representation
        * **b** (ndarray) – optimized linear combination coefficients.

    """
    def __init__(self, mol, dm_tar, pbas=None, Sijt=None, dm_aux=None):
        t = time.time()
        basic(self, mol, pbas, Sijt)

        self.nelec = [int((mol.nelectron+mol.spin)//2), int((mol.nelectron-mol.spin)//2)]
        self.nbas = len(self.Sijt[:, 0, 0])
        self.npot = len(self.Sijt[0, 0, :])
        self.b = np.zeros(2*self.npot)
        self.dm_aux = dm_aux
        self.dm_tar = dm_tar
        self.t_eig = self.t_ws = self.t_gd = self.t_hs = self.t_opt = 0.
        self.t_init = time.time()-t

    get_occ = scf.uhf.get_occ
    make_rdm1 = scf.uhf.UHF.make_rdm1
    spin_square = scf.uhf.UHF.spin_square

    run = run
    time_profile = time_profile
    info = info
    wyscf = wyscf

    def initialize(self):
        """Summary: Construct a fixed part of fock matrix F0

        .. math::

            F_{0} = T + V_{ext} + V_{h} + V_{g}

        """
        self.get_ovlp = lambda *args: self.S
        self.internal_b = np.ones_like(self.b)

        if self.dm_aux is None:
            self.dm_aux = self.dm_tar

        if self.guide is None:
            self.V0 = np.zeros_like(self.dm_tar)
        elif self.guide.lower() == 'faxc':
            N = self.mol.nelectron
            J_tar = scf.hf.get_jk(self.mol, self.dm_aux)[0]
            VFA = ((N-1.)/N)*(J_tar[0]+J_tar[1])
            self.V0 = (VFA, VFA)
        else:
            mydft = dft.UKS(self.mol)
            mydft.xc = self.guide
            self.V0 = mydft.get_veff(self.mol, dm=self.dm_aux)
        self.F0 = (self.T+self.V+self.V0[0],
                   self.T+self.V+self.V0[1])

    def solve(self, b):
        """Summary: Under a given b vector, construct a fock matrix and diagonalize it

        F = F0 + V(b)

        FC = SCE

        resulting mo_coeff(C), mo_energy(E), and density matrix are stored as instance attributes
        """
        if not np.allclose(b, self.internal_b, rtol=1e-12, atol=1e-12):
            t = time.time()
            self.internal_b = b.copy()
            Fa = self.F0[0]+np.einsum('t,ijt->ij', b[:self.npot], self.Sijt)
            Fb = self.F0[1]+np.einsum('t,ijt->ij', b[self.npot:], self.Sijt)
            self.fock = ((Fa, Fb))
            e_a, c_a = scf.hf.eig(Fa, self.S)
            e_b, c_b = scf.hf.eig(Fb, self.S)

            self.mo_coeff = np.array((c_a, c_b))
            self.mo_energy = np.array((e_a, e_b))
            self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
            self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)
            self.t_eig += time.time()-t

            t = time.time()
            if kf_imported:
                self.grad_a = kspies_fort.einsum_ij_ijt_2t((self.dm[0]-self.dm_tar[0]), self.Sijt, self.nbas, self.npot)
                self.grad_b = kspies_fort.einsum_ij_ijt_2t((self.dm[1]-self.dm_tar[1]), self.Sijt, self.nbas, self.npot)
            else:
                self.grad_a = np.einsum('ij,ijt->t', (self.dm[0]-self.dm_tar[0]), self.Sijt)
                self.grad_b = np.einsum('ij,ijt->t', (self.dm[1]-self.dm_tar[1]), self.Sijt)
            self.t_gd += time.time()-t

    def eval_Ws(self, b):
        """Summary: Calculate objective function Ws under given b vector

        When mw.reg > 0, regularization is added
        """
        self.solve(b)
        t = time.time()

        Ws = np.einsum('ij,ji', self.T, (self.dm[0]+self.dm[1]))
        Ws += np.einsum('ij,ji', self.V, (self.dm[0]+self.dm[1]-self.dm_tar[0]-self.dm_tar[1]))
        Ws += np.einsum('ij,ji', self.V0[0], (self.dm[0]-self.dm_tar[0]))
        Ws += np.einsum('ij,ji', self.V0[1], (self.dm[1]-self.dm_tar[1]))
        Ws += np.einsum('t,t', b[:self.npot], self.grad_a)
        Ws += np.einsum('t,t', b[self.npot:], self.grad_b)
        self.Ws = Ws

        penelty = 0.
        if self.reg > 0: #Regularization on
            penelty = self.reg*self.Dvb(b)

        self.t_ws += time.time()-t
        return -(Ws-penelty)

    def eval_Gd(self, b):
        """Summary: -d(Ws)/b_t
        """
        self.solve(b)
        t = time.time()
        self.Ga = self.grad_a
        self.Gb = self.grad_b

        if self.reg > 0: #Regularization on
            self.Ga -= 2*self.reg*np.einsum('st,t->s', self.Tp, b[:self.npot])
            self.Gb -= 2*self.reg*np.einsum('st,t->s', self.Tp, b[self.npot:])

        self.Gd = np.concatenate((self.Ga, self.Gb), axis=None)
        self.t_gd += time.time()-t
        return -self.Gd

    def eval_Hs(self, b):
        """Summary: -d^2(Ws)/(b_t)(b_u)
        """
        self.solve(b)
        t = time.time()

        n_a, n_b = self.nelec[0], self.nelec[1]
        mo_a, mo_b = self.mo_coeff
        eia_a = self.mo_energy[0][:n_a, None] - self.mo_energy[0][None, n_a:]
        eia_b = self.mo_energy[1][:n_b, None] - self.mo_energy[1][None, n_b:]

        if kf_imported:
            self.Ha = .5*kspies_fort.wy_hess(self.Sijt, mo_a, eia_a, n_a, self.nbas-n_a, self.npot)
            self.Hb = .5*kspies_fort.wy_hess(self.Sijt, mo_b, eia_b, n_b, self.nbas-n_b, self.npot)
        else:
            Siat_a = np.einsum('mi,va,mvt->iat', mo_a[:,:n_a], mo_a[:,n_a:], self.Sijt)
            Siat_b = np.einsum('mi,va,mvt->iat', mo_b[:,:n_b], mo_b[:,n_b:], self.Sijt)
            self.Ha = 2*np.einsum('iau,iat,ia->ut', Siat_a, Siat_a, eia_a**-1)
            self.Hb = 2*np.einsum('iau,iat,ia->ut', Siat_b, Siat_b, eia_b**-1)
        self.Hs = np.block([[self.Ha, np.zeros((self.npot, self.npot))],
                            [np.zeros((self.npot, self.npot)), self.Hb]])

        if self.reg > 0:
            self.Hs[self.npot:,self.npot:] -= 2*self.reg*self.Tp
            self.Hs[:self.npot,:self.npot] -= 2*self.reg*self.Tp
        self.t_hs += time.time()-t
        return -self.Hs

    def Dvb(self, b=None):
        """Summary: Calcuate the norm of correction potential derivative

        .. math::

            \\| \\nabla v_C ( \\textbf{r} ) \\| = \\frac{1}{2} \\sum_{\\sigma} \\int v_{C,\\sigma} ( \\textbf{r} ) \\nabla^2 v_{C,\\sigma} ( \\textbf{r} ) d \\textbf{r}

        """
        if b is None:
            ba = self.b[:len(self.b)//2]
            bb = self.b[len(self.b)//2:]
        else:
            ba = b[:len(b)//2]
            bb = b[len(b)//2:]
        Dvb = np.einsum('s,st,t', ba, self.Tp, ba)
        Dvb += np.einsum('s,st,t', bb, self.Tp, bb)
        return Dvb

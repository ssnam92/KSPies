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

    **2020-11-12** guiding potential parser

    **2021-01-07** Analytical three-center overlap integral with auxiliary basis

    **2021-02-08** Support target density matrix in different basis representation

"""

import time, warnings
import numpy as np
from scipy.optimize import minimize
from pyscf import gto, scf, dft, df
from kspies import util
from opt_einsum import contract
try:
    from kspies import kspies_fort
    kf_imported=True
except:
    kf_imported=False
#kf_imported=False

def project_b(mw1, mw2):
    """Summary: Transfer b vector from one WY calculation to other
    """
    mol1 = mw1.pmol
    npbs1 = mol1.nao_nr()
    b1 = mw1.b
    mol2 = mw2.pmol
    if type(mw1).__name__ == 'RWY' and type(mw2).__name__ == 'RWY':
        b1 = np.array([b1]).T
        mw2.b = scf.addons.project_mo_nr2nr(mol1, b1, mol2)[:,0]
    elif type(mw1).__name__ == 'UWY' and type(mw2).__name__ == 'UWY':
        b1 = np.vstack((b1[:npbs1],b1[npbs1:])).T
        b2 = scf.addons.project_mo_nr2nr(mol1, b1, mol2)
        mw2.b = np.hstack((b2[:,0],b2[:,1]))
    else:
        warnings.warn("Two WY objects are not same!")

def numint_3c2b(mol, pbas, level=5):
    """Summary: Three-center overlap integral with different atomic- and potential- basis sets with numerical integration

    Args:
        mol : an instance of :class:`Mole`
        pbas (list) : potential basis set for WY, given as a list of functions 
        level (int 0~9) : PySCF preset mesh grid used for numerical integration. Default is 5

    Returns:
        (ndarray): **Sijt** three-center overlap integral with shape ((n1, n1, n2))
        where n1 is the length of atomic orbital basis set defined in mol,
        and n2 is the length of potential basis set

    """
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    coords = grids.coords
    weights = grids.weights
    ao1 = dft.numint.eval_ao(mol, coords, deriv=0)
    ao2 = []
    n1 = mol.nao_nr()
    n2 = 0

    for vr_generator in pbas:
       n2 += 1
       ao2.append(vr_generator(coords))
    ao2=np.array(ao2).T

    t0 = time.time()
    if kf_imported: #Utilize faster numerical integration
        Sijt = kspies_fort.ovlp_aab(weights, ao1, ao2, n1, n2, len(weights))
    else: #kspies_fort not imported, us numpy
        Sijt = contract('ri, rj, rk, r->ijk', ao1, ao1, ao2, weights)
    n1 = mol.nao_nr()
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
    * Hs_eval    : time to evaluate hessian of objective function

    """
    print("*********Time Profile*********")
    print("initialize : %15.3f"%mw.t_init)
    print("total_opt  : %15.3f"%mw.t_opt)
    print("solve_eig  : %15.3f"%mw.t_eig)
    print("Ws_eval    : %15.3f"%mw.t_ws)
    print("Gd_eval    : %15.3f"%mw.t_gd)
    print("Hs_eval    : %15.3f"%mw.t_hs)

def run(mw, xbas=None):
    """Summary: Minimize an objective function -Ws

    Args:
        mw : RWY or UWY object

    """
    mw.initialize(xbas)
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
        mw.run(xbas=mw.mol.basis)
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

def basic(mw, mol, pbas, Sijt, tbas, Smnt):
    """Summary: Common basic initialization function for RWY and UWY objects

    Args:
        mw : RWY or UWY object
        mol : an instance of :class:`Mole`
        pbas (dict or str) : potential basis set for WY
        Sijt (ndarray) : three-center overlap integral with shape ((no, no, np))
          where no is the length of atomic orbital basis set defined in mol,
          while np is the length of potential basis set
          Only effective for model systems and molecular systems with user-defined pbs.
          Otherwise, this is ignored and analytical integration is performed.
        tbas (dict or str) : basis set of target density matrix
        Smnt (ndarray) : three-center overlap integral with shape ((nt, nt, np))
          where nt is the length of atomic orbital basis set that defines target density matrix,
          while np is the length of potential basis set
          Only effective for model systems and molecular systems with user-defined pbs.
          Otherwise, this is ignored and analytical integration is performed.
    """

    mw.mol = mol
    mw.guide = 'faxc'
    mw.reg = 0.
    mw.tol = 1e-6
    mw.method = 'trust-exact'
    mw.verbose = mw.mol.verbose
    mw.stdout = mw.mol.stdout

    mw.Sijt = None
    mw.Smnt = None
    mw.Tp = None
    mw.pbas = pbas
    mw.tbas = tbas

    is_model1 = len(mw.mol._atm) == 0
    is_model2 = len(mw.mol._bas) == 0
    mw.model = is_model1 and is_model2
    if mw.model: #Check if defined mol is a molecule or model
        mw.guide = None
        if Sijt is None:
            raise AssertionError("Three-center overlap integeral should be given for model system")
        if kf_imported:
            mw.Sijt = np.array(Sijt, order='F')
            if Smnt is not None:
                mw.Smnt = np.array(Smnt, order='F')
        else:
            mw.Sijt = np.array(Sijt, order='C')
            if Smnt is not None:
                mw.Smnt = np.array(Smnt, order='C')
        return

    mw.S = mw.mol.intor_symmetric('int1e_ovlp')
    mw.T = mw.mol.intor_symmetric('int1e_kin')
    mw.V = mw.mol.intor_symmetric('int1e_nuc')

    if tbas is None:
        mw.tbas = mol.basis
    mw.tmol = df.make_auxmol(mol, auxbasis=mw.tbas)

    if pbas is not None and callable(pbas[0]):
        #potential basis is given as a list of user-defined functions
        #In this case, mw.Tp should be given by the user to use regularization
        mw.Sijt = Sijt
        if Sijt is None:
            mw.Sijt = numint_3c2b(mol, pbas)

        if not gto.mole.same_basis_set(mol, mw.tmol):
            mw.Smnt = Smnt
            if Smnt is None:
                mw.Smnt = numint_3c2b(mw.tmol, pbas)

    else:
        #potential basis is given as a pyscf-supported format
        if pbas is None:
            mw.pbas = mol.basis

        mw.pmol = df.make_auxmol(mol, auxbasis=mw.pbas)

        mw.Tp = mw.pmol.intor_symmetric('int1e_kin')
        mw.Sijt = df.incore.aux_e2(mol, mw.pmol, intor='int3c1e')

        if not gto.mole.same_basis_set(mol, mw.tmol):
            mw.Smnt = df.incore.aux_e2(mw.tmol, mw.pmol, intor='int3c1e')

    mw.nbas = len(mw.Sijt[:, 0, 0])
    mw.npot = len(mw.Sijt[0, 0, :])


class RWY:
    """Summary: Perform WY calculation in restricted scheme

    .. _restrictedwy:

    Attributes:
        mol (object) : an instance of :class:`Mole`
        dm_tar (ndarray) : Density matrix of target density in atomic orbital basis representation
        pbas (dict or str or list) : Potential basis set for WY. If not given, same with atomic orbital basis. 
                                     If this is given as a list of function, those functions are recognized as potential basis set.
        Sijt (ndarray) : Three-center overlap integral. Ignored (calculated analitically) when pbas is given as dict or str.
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
    def __init__(self, mol, dm_tar, pbas=None, Sijt=None, tbas=None, Smnt=None, dm_aux=None):
        t = time.time()
        basic(self, mol, pbas, Sijt, tbas, Smnt)

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

    def initialize(self, xbas=None):
        """Summary: Construct a fixed part of fock matrix F0

        .. math::

            F_{0} = T + V_{ext} + V_{h} + V_{g}

        """
        self.get_ovlp = lambda *args: self.S
        self.internal_b = np.ones_like(self.b)

        if self.dm_aux is None:
            self.dm_aux = self.dm_tar

        if xbas is None:
            xbas = self.tbas

        if self.guide is None:
            self.V0 = np.zeros_like(self.dm_tar)
        else:
            fac_faxc, dft_xc = util.parse_guide(self.guide)

            N = self.mol.nelectron

            self.xmol = df.make_auxmol(self.mol, auxbasis=xbas)
            dm_aux_ao = scf.addons.project_dm_nr2nr(self.xmol, self.dm_aux, self.mol)

            J_tar = scf.hf.get_jk(self.mol, dm_aux_ao)[0]
            VFA = -(1./N)*(J_tar)

            mydft = dft.RKS(self.mol)
            mydft.xc = dft_xc
            Vxcdft = mydft.get_veff(self.mol, dm=dm_aux_ao)

            self.V0 = fac_faxc * VFA + Vxcdft

            if self.Smnt is not None:
                #Generate potential matrix in target DM basis representation
                dm_aux_tbas = scf.addons.project_dm_nr2nr(self.xmol, self.dm_aux, self.tmol)
 
                self.V_tbas = self.tmol.intor('int1e_nuc')
                VFA_tbas = -(1./N)*scf.hf.get_jk(self.tmol, dm_aux_tbas)[0]
                mydft_tbas = dft.RKS(self.tmol)
                mydft_tbas.xc = dft_xc
                Vxcdft_tbas = mydft_tbas.get_veff(self.tmol, dm=dm_aux_tbas)
                self.V0_tbas = fac_faxc * VFA_tbas + Vxcdft_tbas

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
                if self.Smnt is None:
                    #AO basis for WY is same with AO basis of target density
                    self.grad = kspies_fort.einsum_ij_ijt_2t((self.dm-self.dm_tar), self.Sijt, self.nbas, self.npot)
                else:
                    self.grad = kspies_fort.einsum_ij_ijt_2t(self.dm, self.Sijt, self.nbas, self.npot)
                    self.grad -= kspies_fort.einsum_ij_ijt_2t(self.dm_tar, self.Smnt, len(self.Smnt[:, 0, 0]), self.npot)
            else:
                if self.Smnt is None:
                    self.grad = contract('ij,ijt->t', (self.dm-self.dm_tar), self.Sijt)
                else:
                    self.grad = contract('ij,ijt->t', self.dm, self.Sijt)
                    self.grad -= contract('ij,ijt->t', self.dm_tar, self.Smnt)

            self.t_gd += time.time()-t

    def eval_Ws(self, b):
        """Summary: Calculate objective function Ws under given b vector

        When mw.reg > 0, regularization is added
        """
        self.solve(b)
        t = time.time()

        Ws = np.einsum('ij,ji', self.T, self.dm)

        if self.Smnt is None:
            Ws += np.einsum('ij,ji', (self.V+self.V0), (self.dm-self.dm_tar))
        else:
            Ws += np.einsum('ij,ji', (self.V+self.V0), self.dm)
            Ws -= np.einsum('ij,ji', (self.V_tbas+self.V0_tbas), self.dm_tar)
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

        tol_zero = 1e+15
        with np.errstate(divide='ignore'):
            eia = np.nan_to_num(eia**-1, posinf=tol_zero, neginf=-tol_zero)**-1

        if kf_imported:
            self.Hs=kspies_fort.wy_hess(self.Sijt, self.mo_coeff, eia, nocc, self.nbas-nocc, self.npot)
        else:
            Siat = contract('mi,va,mvt->iat', self.mo_coeff[:,:nocc], self.mo_coeff[:,nocc:], self.Sijt)
            self.Hs= 4*contract('iau,iat,ia->ut', Siat, Siat, eia**-1)

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
        Dvb = 2*np.einsum('s,st,t', b, self.Tp, b, optimize=True)
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
    def __init__(self, mol, dm_tar, pbas=None, Sijt=None, tbas=None, Smnt=None, dm_aux=None):
        t = time.time()
        basic(self, mol, pbas, Sijt, tbas, Smnt)

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

    def initialize(self, xbas=None):
        """Summary: Construct a fixed part of fock matrix F0

        .. math::

            F_{0} = T + V_{ext} + V_{h} + V_{g}

        """
        self.get_ovlp = lambda *args: self.S
        self.internal_b = np.ones_like(self.b)

        if self.dm_aux is None:
            self.dm_aux = self.dm_tar
        if xbas is None:
            xbas = self.tbas

        if self.guide is None:
            self.V0 = np.zeros_like(self.dm_tar)
        else:
            fac_faxc, dft_xc = util.parse_guide(self.guide)

            N = self.mol.nelectron
            self.xmol = df.make_auxmol(self.mol, auxbasis=xbas)
            dm_aux_ao = scf.addons.project_dm_nr2nr(self.xmol, self.dm_aux, self.mol)

            J_tar = scf.hf.get_jk(self.mol, dm_aux_ao)[0]
            VFA = -(1./N)*(J_tar[0]+J_tar[1])

            mydft = dft.UKS(self.mol)
            mydft.xc = dft_xc
            Vxcdft = mydft.get_veff(self.mol, dm=dm_aux_ao)

            if self.Smnt is not None:
                #Generate potential matrix in target DM basis representation
                dm_aux_tbas = scf.addons.project_dm_nr2nr(self.xmol, self.dm_aux, self.tmol)

                self.V_tbas = self.tmol.intor('int1e_nuc')
                J_tbas = scf.hf.get_jk(self.tmol, dm_aux_tbas)[0]
                VFA_tbas = -(1./N)*(J_tbas[0]+J_tbas[1])
                mydft_tbas = dft.UKS(self.tmol)
                mydft_tbas.xc = dft_xc
                Vxcdft_tbas = mydft_tbas.get_veff(self.tmol, dm=dm_aux_tbas)
                self.V0_tbas = fac_faxc * VFA_tbas + Vxcdft_tbas

            self.V0 = fac_faxc * np.array((VFA, VFA)) + Vxcdft

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
            Fa = self.F0[0]+contract('t,ijt->ij', b[:self.npot], self.Sijt)
            Fb = self.F0[1]+contract('t,ijt->ij', b[self.npot:], self.Sijt)
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
                if self.Smnt is None:
                    #AO basis for WY is same with AO basis of target density
                    self.grad_a = kspies_fort.einsum_ij_ijt_2t((self.dm[0]-self.dm_tar[0]), self.Sijt, self.nbas, self.npot)
                    self.grad_b = kspies_fort.einsum_ij_ijt_2t((self.dm[1]-self.dm_tar[1]), self.Sijt, self.nbas, self.npot)
                else:
                    self.grad_a = kspies_fort.einsum_ij_ijt_2t(self.dm[0], self.Sijt, self.nbas, self.npot)
                    self.grad_a -= kspies_fort.einsum_ij_ijt_2t(self.dm_tar[0], self.Smnt, len(self.Smnt[:, 0, 0]), self.npot)
                    self.grad_b = kspies_fort.einsum_ij_ijt_2t(self.dm[1], self.Sijt, self.nbas, self.npot)
                    self.grad_b -= kspies_fort.einsum_ij_ijt_2t(self.dm_tar[1], self.Smnt, len(self.Smnt[:, 0, 0]), self.npot)
            else:
                if self.Smnt is None:
                    self.grad_a = contract('ij,ijt->t', (self.dm[0]-self.dm_tar[0]), self.Sijt)
                    self.grad_b = contract('ij,ijt->t', (self.dm[1]-self.dm_tar[1]), self.Sijt)
                else:
                    self.grad_a = contract('ij,ijt->t', self.dm[0], self.Sijt)
                    self.grad_a -= contract('ij,ijt->t', self.dm_tar[0], self.Smnt)
                    self.grad_b = contract('ij,ijt->t', self.dm[1], self.Sijt)
                    self.grad_b -= contract('ij,ijt->t', self.dm_tar[1], self.Smnt)

            self.t_gd += time.time()-t

    def eval_Ws(self, b):
        """Summary: Calculate objective function Ws under given b vector

        When mw.reg > 0, regularization is added
        """
        self.solve(b)
        t = time.time()

        Ws = np.einsum('ij,ji', self.T, (self.dm[0]+self.dm[1]))

        if self.Smnt is None:
            Ws += np.einsum('ij,ji', self.V, (self.dm[0]+self.dm[1]-self.dm_tar[0]-self.dm_tar[1]))
            Ws += np.einsum('ij,ji', self.V0[0], (self.dm[0]-self.dm_tar[0]))
            Ws += np.einsum('ij,ji', self.V0[1], (self.dm[1]-self.dm_tar[1]))

        else:
            Ws += np.einsum('ij,ji', self.V, (self.dm[0]+self.dm[1]))
            Ws -= np.einsum('ij,ji', self.V_tbas, (self.dm_tar[0]+self.dm_tar[1]))
            Ws += np.einsum('ij,ji', self.V0[0], self.dm[0])
            Ws -= np.einsum('ij,ji', self.V0_tbas[0], self.dm_tar[0])
            Ws += np.einsum('ij,ji', self.V0[1], self.dm[1])
            Ws -= np.einsum('ij,ji', self.V0_tbas[1], self.dm_tar[1])

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

        tol_zero = 1e+15
        with np.errstate(divide='ignore'):
            eia_a = np.nan_to_num(eia_a**-1, posinf=tol_zero, neginf=-tol_zero)**-1
            eia_b = np.nan_to_num(eia_b**-1, posinf=tol_zero, neginf=-tol_zero)**-1

        if kf_imported:
            self.Ha = .5*kspies_fort.wy_hess(self.Sijt, mo_a, eia_a, n_a, self.nbas-n_a, self.npot)
            self.Hb = .5*kspies_fort.wy_hess(self.Sijt, mo_b, eia_b, n_b, self.nbas-n_b, self.npot)
        else:
            Siat_a = contract('mi,va,mvt->iat', mo_a[:,:n_a], mo_a[:,n_a:], self.Sijt)
            Siat_b = contract('mi,va,mvt->iat', mo_b[:,:n_b], mo_b[:,n_b:], self.Sijt)
            self.Ha = 2*contract('iau,iat,ia->ut', Siat_a, Siat_a, eia_a**-1)
            self.Hb = 2*contract('iau,iat,ia->ut', Siat_b, Siat_b, eia_b**-1)
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
        Dvb = contract('s,st,t', ba, self.Tp, ba)
        Dvb += contract('s,st,t', bb, self.Tp, bb)
        return Dvb

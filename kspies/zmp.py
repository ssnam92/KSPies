"""
Zhao-Morrison-Parr
==================

**Summary** This script preforms a Zhao-Morrison-Parr [ZMP1994]_ Kohn Sham inversion.

Written for Python 3.7.4

  :References:

    .. [ZMP1994] Qingsheng Zhao, Robert C Morrison, and Robert G Parr.
        From electron densities to Kohn-Sham kinetic energies, orbital energies, exchange-correlation potentials, and exchange-correlation energies. (1994)
        <https://doi.org/10.1103/PhysRevA.50.2138> Physical Review A, 50(3) 2138.

    .. [THG1997] David J Tozer, Nicholas C Handy, and William H Green.
        Exchange-correlation functionals from ab initio electron densities (1997)
        <https://doi.org/10.1016/S0009-2614(97)00586 and-1> Chemical Physics Letters, 273(3-4) 183-194

    .. [Pulay1980] P Pulay.
        Convergence acceleration of iterative sequences. the case of SCF iteration (1980)
        <https://doi.org/10.1016/0009-2614(80)80396-4> Chemical Physics Letters, 73 (2): 393–398

.. moduleauthor::
    Seungsoo Nam <skaclitz@yonsei.ac.kr> <http://tccl.yonsei.ac.kr/mediawiki/index.php/Main_Page> ORCID: `000-0001-9948-6140 <https://orcid.org/0000-0001-9948-6140>`_

.. topic:: Funding

    This research was made possible by funding from the National Research Foundation of Korea (NRF-2020R1A2C2007468 and NRF-2020R1A4A1017737).

.. topic:: Internal Log

    **2020-06-13** RJM added doc string templates

    **2020-08-21** SN corrected typos, minor changes in attribute names, etc.

"""

from functools import reduce
import numpy as np
from pyscf import scf, dft


class DIIS:
    """Summary: Class for DIIS extrapolation used in ZMP

       See [Pulay1980]_ for some extra context.

    """
    def __init__(self, S, diis_space):
        """Initialize DIIS object

        Args:
            S (ndarray):  overlap integral
            diis_space (integer) : number of DIIS vectors used in extrapolation

        """
        eig, Z = np.linalg.eigh(S)
        S12 = 1./np.sqrt(eig)
        self.S = S
        self.O = reduce(np.dot, (Z, np.diag(S12), Z.T))
        self.diis_space = diis_space
        self.norb = len(S[0])
        self.ems = np.zeros((self.diis_space, self.norb, self.norb))
        self.pms = np.zeros((self.diis_space, self.norb, self.norb))
        self.tall = self.t_1 = self.t_2 = self.t_3 = 0.

    def extrapolate(self, iteration, fock, dm):
        """Summary: New fock matrix by linear combination of previous fock matrices

        Args:
            iteration (integer) : present SCF iteration
            fock (ndarray) : fock matrix
            dm (ndarray) : density matrix obtained from previous step

        Returns:
            (tuple): tuple containing:

                (ndarray): **newfock** extrapolated fock matrix

                (float): **diis_error** DIIS error used in convergence criteria

        """

        if iteration <= 1 or self.diis_space < 2:
            return fock, 0.0

        for k in range(1, min(iteration, self.diis_space))[::-1]:
            self.ems[k] = self.ems[k-1]
            self.pms[k] = self.pms[k-1]

        em = reduce(np.dot, (fock, dm, self.S))
        em -= em.T
        self.ems[0] = reduce(np.dot, (self.O.T, em, self.O))
        self.pms[0] = fock[:]
        idx = np.abs(self.ems[0]).argmax()
        diis_error = np.abs(np.ravel(self.ems[0])[idx])

        #Solve BC = A to find C
        nb = min(iteration, self.diis_space)-1
        B = -1.*np.ones((nb+1, nb+1))
        B[nb, nb] = 0.
        B[:nb, :nb] = np.einsum('aij,bji->ab', self.ems[:nb, :, :], self.ems[:nb, :, :],optimize='greedy')
        A = np.zeros(nb+1)
        A[nb] = -1.
        C = np.linalg.solve(B, A)

        # form new extrapolated diis fock matrix
        newfock = np.zeros_like(fock)
        for i, c in enumerate(C[:-1]):
            newfock += c*self.pms[i]

        return newfock, diis_error

def basic(mz, mol):
    """Summary: Common basic initialization function for RZMP and UZMP objects

    Args:
        mz : RZMP or UZMP object
        mol (object) : an instance of :class:`Mole`

    """
    mz.mol = mol
    mz.guide = 'faxc'
    mz.diis_space = 40
    mz.level_shift = .2
    mz.max_cycle = 400
    mz.conv_tol_dm = 1e-7
    mz.conv_tol_diis = 1e-5
    mz.with_df = False
    mz.verbose = mz.mol.verbose
    mz.stdout = mz.mol.stdout

    mz.S = mz.mol.intor_symmetric('int1e_ovlp')
    mz.T = mz.mol.intor_symmetric('int1e_kin')
    mz.V = mz.mol.intor_symmetric('int1e_nuc')

class RZMP:
    """Summary: Perform ZMP calculation in restricted scheme, see [ZMP1994]_ for detail.

    .. _restricted-zmp:

    Attributes:
        mol (object) : an instance of :class:`Mole`
        dm_tar (ndarray) : Density matrix of target density in atomic orbital basis representation
        dm_aux (ndarray) : Auxilary density matrix to construct a fixed part of fock matrix. Default is dm_tar
        guide (str) : Guiding potential. Can be set as

            |  None : no guiding potential except external potential
            |  'faxc' : Exchange-correlation part of Fermi-Amaldi potential
            |  xc   : ks.xc attribute in pyscf DFT

        diis_space (int) : DIIS space size. Default is 40
        level_shift (float) : Level shift (in AU) for virtual space. Default is 0.2
        max_cycle (int) : max number of zscf iterations. Defalut is 400
        conv_tol_dm (float) :  converge threshold for density matrix. Default is 1e-7
        conv_tol_diis (float) : converge threshold for DIIS error. Default is 1e-5

    :ivar:
        * **converged** (bool) – zscf converged or not
        * **mo_energy** (ndarray) –  molecular orbital energies (Note that energies when level_shift = 0)
        * **mo_occ** (ndarray) –  molecular orbital occupation numbers
        * **mo_coeff** (ndarray) – molecular orbital coefficients
        * **dm** (ndarray) – density matrix in atomic orbital basis representation
        * **l** (float) – the last given lambda

    """
    def __init__(self, mol, dm_tar, dm_aux=None):
        basic(self,mol)

        self.dm_tar = dm_tar
        self.initialized = False
        self.dm = dm_tar
        self.dm_aux = dm_aux
        self.dm_old = dm_tar
        self.verbose = mol.verbose

    get_occ = scf.hf.get_occ
    make_rdm1 = scf.hf.SCF.make_rdm1

    def initialize(self):
        """Summary: Construct a fixed part of fock matrix F0

        .. math::

            F_{0} = T + V_{ext} + V_{h} + V_{g}

        """
        self.get_ovlp=lambda *args: self.S

        if self.with_df:
            self.mf=dft.RKS(self.mol).density_fit()
        else:
            self.mf=dft.RKS(self.mol)
        self.mf.grids.build()
        self.coords=self.mf.grids.coords
        self.weights=self.mf.grids.weights
        self.ao=dft.numint.eval_ao(self.mol,self.coords)

        if self.dm_aux is None :
            self.dm_aux=self.dm_tar
        self.J_tar=self.mf.get_jk(self.mol,self.dm_tar)[0]

        if self.guide is None:
            self.V0=np.zeros_like(self.dm_tar)
        elif self.guide.lower()=='faxc':
            N=self.mol.nelectron
            self.J_aux=self.mf.get_jk(self.mol,self.dm_aux)[0]
            self.V0=((N-1.)/N)*(self.J_aux)
        else:
            self.mf.xc=self.guide
            self.V0=self.mf.get_veff(self.mol,dm=self.dm_aux)
        self.F0=self.T+self.V+self.V0
        self.initialized=True

    def zscf(self, l):
        """Summary:
            Run self-consistent ZMP equation under given lambda (l). Prints lambda,
            HOMO-LUMO, dN and C to terminal.

            Args:
                l (float): Lagrange multiplier lambda
        """
        if not self.initialized:
            self.initialize()
        self.l = l

        self.converged = False
        self.zdiis = DIIS(self.S, self.diis_space)

        for cycle in range(1, self.max_cycle):
            self.J = self.mf.get_jk(self.mol, self.dm)[0]

            self.F = self.F0 + l*(self.J - self.J_tar)
            self.F = scf.hf.level_shift(self.S, self.dm*.5, self.F, self.level_shift)
            self.F, diis_e = self.zdiis.extrapolate(cycle, self.F, self.dm) #DIIS

            self.mo_energy, self.mo_coeff = scf.hf.eig(self.F, self.S)
            self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
            self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)

            ddm = self.dm_old - self.dm
            dm_e = np.max(np.abs(ddm))
            self.dm_old = self.dm
            dm_converged = dm_e < self.conv_tol_dm
            diis_converged = diis_e < self.conv_tol_diis
            self.mo_energy[self.mo_occ==0] -= self.level_shift

            nocc = self.mol.nelectron // 2
            HOMO, LUMO =self.mo_energy[nocc-1], self.mo_energy[nocc]
            gap = LUMO - HOMO

            print(f'\rlambda= {l:7.2f}  iter: {cycle:4d} gap= {gap:10.7f}   ',end='\r')

            self.converged = dm_converged and diis_converged
            if self.converged and cycle > 1:
                break

        self.J = self.mf.get_jk(self.mol, self.dm)[0]
        dn = dft.numint.eval_rho(self.mol, self.ao, self.dm - self.dm_tar)
        dN = 1000*np.einsum('r,r', abs(dn), self.weights)
        C = np.einsum('ij,ji', self.dm-self.dm_tar, self.J - self.J_tar) #Objective of minimization
        print(f'lambda= {l:7.2f} niter: {cycle:4d} gap= {LUMO-HOMO:10.7f} dN= {dN:7.2f} C= {C:.2e} ')

class UZMP:
    """Summary: Perform ZMP calculation in unrestricted scheme, see [THG1997]_.

    .. _unrestricted-zmp:

    Attributes:
        mol (object) : an instance of :class:`Mole`
        dm_tar (ndarray) : Density matrix of target density in atomic orbital basis representation
        dm_aux (ndarray) : Auxilary density matrix to construct a fixed part of fock matrix. Default is dm_tar
        guide (str) : Guiding potential. Can be set as

            |  None : no guiding potential except external potential
            | 'faxc' : Exchange-correlation part of Fermi-Amaldi potential
            |  xc   : ks.xc attribute in pyscf DFT

        diis_space (int) : DIIS space size. Default is 40
        level_shift (float) : Level shift (in AU) for virtual space. Default is 0.2
        max_cycle (int) : max number of zscf iterations. Defalut is 400
        conv_tol_dm (float) :  converge threshold for density matrix. Default is 1e-7
        conv_tol_diis (float) : converge threshold for DIIS error. Default is 1e-5

    :ivar:
        * **converged** (bool) – zscf converged or not
        * **mo_energy** (ndarray) –  molecular orbital energies (Note that energies when level_shift = 0)
        * **mo_occ** (ndarray) –  molecular orbital occupation numbers
        * **mo_coeff** (ndarray) – molecular orbital coefficients
        * **dm** (ndarray) – density matrix in atomic orbital basis representation
        * **l** (float) – the last given lambda

    """
    def __init__(self, mol, dm_tar, dm_aux=None):
        basic(self, mol)
        self.nelec = [int((mol.nelectron + mol.spin)//2), int((mol.nelectron - mol.spin)//2)]

        self.dm_tar = dm_tar
        self.initialized = False
        self.dm = dm_tar
        self.dm_aux = dm_aux
        self.dm_old = dm_tar
        self.verbose = mol.verbose

    get_occ = scf.uhf.get_occ
    make_rdm1 = scf.uhf.UHF.make_rdm1
    spin_square = scf.uhf.UHF.spin_square

    def initialize(self):
        """Summary: Construct a fixed part of fock matrix F0

        .. math::

            F_{0} = T + V_{ext} + V_{h} + V_{g}

        """
        self.get_ovlp=lambda *args: self.S

        if self.with_df:
            self.mf = dft.UKS(self.mol).density_fit()
        else:
            self.mf = dft.UKS(self.mol)
        self.mf.grids.build()
        self.coords = self.mf.grids.coords
        self.weights = self.mf.grids.weights
        self.ao = dft.numint.eval_ao(self.mol,self.coords)

        if self.dm_aux is None:
            self.dm_aux = self.dm_tar
        self.J_tar = self.mf.get_jk(self.mol,self.dm_tar)[0]

        if self.guide is None:
            self.V0 = np.zeros_like(self.dm_tar)
        elif self.guide.lower() == 'faxc':
            N = self.mol.nelectron
            self.J_aux = self.mf.get_jk(self.mol,self.dm_aux)[0]
            VFA = ((N-1.)/N)*(self.J_aux[0]+self.J_aux[1])
            self.V0 = (VFA,VFA)
        else :
            self.mf.xc = self.guide
            self.V0 = self.mf.get_veff(self.mol, dm=self.dm_aux)
        self.F0 = (self.T + self.V+self.V0[0],
                 self.T + self.V+self.V0[1])
        self.initialized = True

    def zscf(self, l):
        """Summary: Run self-consistent ZMP equation under given lambda (l)

            Args:
            l (float): Lagrange multiplier lambda

        """
        if not self.initialized:
            self.initialize()
        self.l = l

        self.converged = False
        self.zdiis_a = DIIS(self.S, self.diis_space)
        self.zdiis_b = DIIS(self.S, self.diis_space)

        for cycle in range(1, self.max_cycle):
            self.J = self.mf.get_jk(self.mol, self.dm)[0]

            self.Fa = self.F0[0] + 2*l*(self.J[0] - self.J_tar[0])
            self.Fb = self.F0[1] + 2*l*(self.J[1] - self.J_tar[1])

            self.Fa = scf.hf.level_shift(self.S, self.dm[0], self.Fa, self.level_shift)
            self.Fb = scf.hf.level_shift(self.S, self.dm[1], self.Fb, self.level_shift)

            self.Fa, diis_e_a = self.zdiis_a.extrapolate(cycle, self.Fa, self.dm[0])
            self.Fb, diis_e_b = self.zdiis_b.extrapolate(cycle, self.Fb, self.dm[1])

            e_a, c_a = scf.hf.eig(self.Fa, self.S)
            e_b, c_b = scf.hf.eig(self.Fb, self.S)
            self.mo_energy = np.array((e_a,e_b))
            self.mo_coeff = np.array((c_a,c_b))

            self.mo_occ = self.get_occ(self.mo_energy, self.mo_coeff)
            self.dm = self.make_rdm1(self.mo_coeff, self.mo_occ)

            ddm = self.dm_old - self.dm
            dm_e = np.max(np.abs(ddm))
            self.dm_old = self.dm
            dm_converged = dm_e < self.conv_tol_dm
            diis_converged = diis_e_a+diis_e_b < self.conv_tol_diis
            self.mo_energy[0][self.mo_occ[0]==0] -= self.level_shift
            self.mo_energy[1][self.mo_occ[1]==0] -= self.level_shift

            HOMO = np.maximum(self.mo_energy[0][self.nelec[0]-1],
                              self.mo_energy[1][self.nelec[1]-1])
            LUMO = np.minimum(self.mo_energy[0][self.nelec[0]],
                              self.mo_energy[1][self.nelec[1]])
            gap = LUMO-HOMO

            print(f'\rlambda= {l:7.2f}  iter: {cycle:4d} gap= {gap:10.7f}   ',end='\r')

            self.converged = dm_converged and diis_converged
            if self.converged and cycle > 1:
                break

        self.J = self.mf.get_jk(self.mol, self.dm)[0]
        #Calculate alpha/beta density difference separately
        #dn_a = dft.numint.eval_rho(self.mol, self.ao, (self.dm-self.dm_tar)[0])
        #dn_b = dft.numint.eval_rho(self.mol, self.ao, (self.dm-self.dm_tar)[1])
        #dN = 1000*np.einsum('r,r', abs(dn_a)+abs(dn_b), self.weights)
        dn = dft.numint.eval_rho(self.mol, self.ao, (self.dm[0]+self.dm[1]-self.dm_tar[0]-self.dm_tar[1]))
        dN = 1000*np.einsum('r,r', abs(dn), self.weights)

        Ca = np.einsum('ij,ji', self.dm[0]-self.dm_tar[0], self.J[0]-self.J_tar[0])
        Cb = np.einsum('ij,ji', self.dm[1]-self.dm_tar[1], self.J[1]-self.J_tar[1])
        C = 2*(Ca+Cb)
        print(f'lambda= {l:7.2f} niter: {cycle:4d} gap= {gap:10.7f} dN= {dN:7.2f} C= {C:.2e}  ')


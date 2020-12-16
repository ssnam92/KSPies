# -*- coding: utf-8 -*-
"""
Utility Functions
=================

**Summary** Utility functions for KSPies.

  :References:

    .. [Mirko2014] Mirko Franchini, Pierre Herman Theodoor Philipsen, Erik van Lenthe, and Lucas Visscher.
        Accurate Coulomb Potentials for Periodic and Molecular Systems through Density Fitting. (2014)
        <https://doi.org/10.1021/ct500172n> Journal of Chemical Theory and Computation, 10(5), 1994-2004.

.. moduleauthor::
    Seungsoo Nam <skaclitz@yonsei.ac.kr> <http://tccl.yonsei.ac.kr/mediawiki/index.php/Main_Page> ORCID: `000-0001-9948-6140 <https://orcid.org/0000-0001-9948-6140>`_
    Hansol Park <hansol7954@yonsei.ac.kr> <http://tccl.yonsei.ac.kr/mediawiki/index.php/Main_Page> ORCID: `000-0001-8706-5015 <https://orcid.org/0000-0001-8706-5015>`_

.. topic:: functions

    mo2ao

    wfnreader

    eval_vh

    eval_vxc

.. todo::

    * Make IF statment for kspies_fort

.. topic:: Internal Log

    **2020-06-06** SN made edits

    **2020-07-26** Updates on eval_vh (Hansol ver.)

    **2020-08-21** SN corrected typos, minor changes in attribute names, etc.

    **2020-11-11** Added wfnreader

"""
from functools import reduce
import numpy as np
import warnings

from scipy.special import sph_harm
from scipy.spatial import distance_matrix
from scipy.interpolate import CubicSpline
from pyscf import gto, dft, scf
from pyscf.dft import numint
try:
    from kspies import kspies_fort
    kf_imported=True
except:
    kf_imported=False

def mo2ao(mo, p1, p2=None):
    """Summary: Convert mo-basis density matrices to basis-set representation density matrices

        Args:
            mo (ndarray) : molecular orbital coefficients
            p1 (ndarray) : mo-basis one-particle density matrices
            p2 (ndarray) : mo-basis two-particle density matrices, optional

        Returns:
            (tuple): tuple containing:

                (ndarray): **dm1** ao-basis one-particle density matrices

                (ndarray): **dm2** ao-basis two-particle density matrices, returned only when p2 is given
    """
    def _convert_rdm1(mo, p1):
        """ Summary: Convert mo-basis 1-rdm p1 to ao-basis
        """
        return reduce(np.dot, (mo, p1, mo.T))
    def _convert_rdm2(mo1, mo2, p2):
        """ Summary: Convert mo-basis 2-rdm p2 to ao-basis
        """
        nmo = mo1.shape[1]
        p = np.dot(mo1, p2.reshape(nmo, -1))
        p = np.dot(p.reshape(-1, nmo), mo2.T)
        p = p.reshape([nmo]*4).transpose(2, 3, 0, 1)
        p = np.dot(mo2, p.reshape(nmo, -1))
        p = np.dot(p.reshape(-1, nmo), mo1.T)
        p = p.reshape([nmo]*4)
        return p

    if np.array(p1).ndim == 2: #RHF
        dm1 = _convert_rdm1(mo, p1)
        if p2 is None:
            return dm1
        dm2 = _convert_rdm2(mo, mo, p2)
        return dm1, dm2
    elif np.array(p1).ndim == 3:
        if np.array(mo).ndim == 2: #ROHF
            mo_a = mo
            mo_b = mo
        else: #UHF
            mo_a, mo_b = mo
        dm1a = _convert_rdm1(mo_a, p1[0])
        dm1b = _convert_rdm1(mo_b, p1[1])
        if p2 is None:
            return (dm1a, dm1b)
        dm2aa = _convert_rdm2(mo_a, mo_a, p2[0])
        dm2ab = _convert_rdm2(mo_a, mo_b, p2[1])
        dm2bb = _convert_rdm2(mo_b, mo_b, p2[2])
        return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)

TYPE_MAP = [
    [1],  # S
    [2, 3, 4],  # P
    [5, 8, 9, 6, 10, 7],  # D
    [11,14,15,17,20,18,12,16,19,13],  # F
    [21,24,25,30,33,31,26,34,35,28,22,27,32,29,23],  # G
    [56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36],  # H
]

def readwfn(filename, mol, makerdm=False):
    """Summary: Convert wfn file to PySCF format

        Args:
            filename (string) : name of wfn file to convert
            mol (object) : an instance of :class:`Mole`

        Returns:
            (tuple): tuple containing:

                (ndarray): **mo_coeff** molecular orbital coefficient from wfn

                (ndarray): **mo_occ** molecular orbital occupation number from wfn

                (ndarray): **mo_energy** molecular orbital energy from wfn

    """

    if mol.cart:
        raise NotImplementedError('Cartesian basis not available')
      
    with open(filename, 'r') as f:
        f.readline()
        dat = f.readline().split()
        norb = int(dat[-7])
        nprm = int(dat[-4])
        natm = int(dat[-2])
        achrg = [] #For sanity check
        coord = []
        for i in range(natm):
            dat = f.readline().split()
            achrg.append(float(dat[-1]))
            coord.append([float(c) for c in dat[4:7]])
        chrg_criteria = sum(abs(mol.atom_charges() - np.array(achrg))) > 1e-10
        coord_criteria = np.linalg.norm(mol.atom_coords() - np.array(coord)) >  1e-6
        if chrg_criteria or coord_criteria:
            warnings.warn("Different molecule!")

        centr = []
        typea = []
        expos = []
        for i in range(nprm//20+1):
            centr += [int(a) for a in f.readline().split()[2:]]
        for i in range(nprm//20+1):
            typea += [int(a) for a in f.readline().split()[2:]]
        for i in range(nprm//5+1):
            expos += [float(a.replace('D', 'E')) for a in f.readline().split()[1:]]

        MOs = []
        occ = []
        eng = []
        for n in range(norb):
            dat = f.readline().split()
            eng.append(float(dat[-1]))
            occ.append(float(dat[-5]))
            orb = []
            for i in range(nprm//5+1):
                orb += [float(a.replace('D','E')) for a in f.readline().split()]
            MOs.append(orb)
    MOs = np.array(MOs).T

    c2s=[]
    for l in range(5):
        c2s.append(np.linalg.pinv(gto.cart2sph(l)))
    from pyscf.x2c import x2c
    uncmol, ctr = x2c._uncontract_mol(mol, True, 0.)
    uncmol_info = []
    for ib in range(uncmol.nbas):
        ia = uncmol.bas_atom(ib)
        l = uncmol.bas_angular(ib)
        es = uncmol.bas_exp(ib)
        uncmol_info.append([ia+1, l, es[0]])

    match = []
    for ip in range(nprm):
        am = [l for l in range(5) if typea[ip] in TYPE_MAP[l]]
        match.append(uncmol_info.index([centr[ip], am, expos[ip]]))

    Rot = np.zeros((uncmol.nao_nr(), nprm))
    bidx = 0
    for ib in range(uncmol.nbas):
        l = uncmol.bas_angular(ib)
        indices = [i for i, ip2b in enumerate(match) if ip2b==ib]
        matchtype = [typea[idx] for idx in indices]
        reorder = [TYPE_MAP[l].index(i) for i in matchtype]
        trans = c2s[l]*1./float(uncmol._libcint_ctr_coeff(ib))
        Rot[bidx:bidx+2*l+1, indices] = trans[:, reorder]
        bidx += 2*l+1

    #Outputs
    cof = np.linalg.pinv(ctr)@Rot@MOs
    eng = np.array(eng)
    occ = np.array(occ)

    restri = False
    isvirt = False
    if max(occ) > 1.3: 
        #Assuming NO occupation numbers of unrestricted calculation 
        #does not exceed 1.3
        restri = True
    if restri and norb > mol.nelectron//2:
        isvirt = True
    elif not restri and norb > mol.nelectron:
        isvirt = True

    s1e = mol.intor_symmetric('int1e_ovlp')
    if restri:
        if isvirt:
            mo_occ = occ
            mo_energy = eng
            mo_coeff = cof
        else:
            mo_occ = np.zeros((mol.nao_nr()))
            mo_energy = np.zeros((mol.nao_nr()))
            mo_coeff = np.zeros((mol.nao_nr(), mol.nao_nr()))
            mo_occ[:len(occ)] = occ
            mo_energy[:len(eng)] = eng
            mo_coeff[:, :len(occ)] = cof
        chk = np.einsum('ki,kl,lj->ij', cof, s1e, cof, optimize='greedy')
        condi = np.linalg.norm(chk - np.eye(len(chk[0])))
        if makerdm:
            dm = scf.hf.make_rdm1(mo_coeff, mo_occ)
    else: #assuming orbital order (alpha_0, alpha_1, ... beta_0, beta_1 ...)
        if isvirt:
            na = norb//2
            mo_occ = np.array([occ[:na], occ[na:]])
            mo_energy = np.array([eng[:na], eng[na:]])
            mo_coeff = np.array([cof[:, :na], cof[:, na:]])
        else:
            na, nb = mol.nelec
            if not na+nb == norb:
                warnings.warn("Proper number of electron should give from Mole object")
                return cof, occ, eng
            mo_occ = np.zeros((2, mol.nao_nr()))
            mo_energy = np.zeros((2, mol.nao_nr()))
            mo_coeff = np.zeros((2, mol.nao_nr(), mol.nao_nr()))
            mo_occ[0, :na] = occ[:na]
            mo_occ[1, :nb] = occ[na:]
            mo_energy[0, :na] = eng[:na]
            mo_energy[1, :nb] = eng[na:]
            mo_coeff[0,: ,:na] = cof[:, :na]
            mo_coeff[1,: ,:nb] = cof[:, na:]
        chk_a = np.einsum('ki,kl,lj->ij', cof[:, :na], s1e, cof[:, :na], optimize='greedy')
        chk_b = np.einsum('ki,kl,lj->ij', cof[:, na:], s1e, cof[:, na:], optimize='greedy')
        condi = np.linalg.norm(chk_a - np.eye(len(chk_a[0])))
        condi += np.linalg.norm(chk_b - np.eye(len(chk_b[0])))
        if makerdm:
            dm = scf.uhf.make_rdm1(mo_coeff, mo_occ)

    if condi > 1e-5:
        print("Orthonrmal conditonal number:", condi, "> 1e-5")
        warnings.warn("Converted MOs are not orthonormal")
    if makerdm:
        return mo_coeff, mo_occ, mo_energy, dm
    else:
        return mo_coeff, mo_occ, mo_energy


def parse_guide(description):
    """Summary: Guiding potential parser for ZMP and WY

        Args:
            description (str) : guiding potential description for inversion

        Returns:
            (tuple): tuple containing:

                (float): **fac_faxc** factor for Fermi-Amaldi potential (faxc)

                (string): **dft_xc** description of dft part of xc

    """
    def _parse_guide(description):
        fac_faxc = 0
        dftxc = ''
        for token in description.replace('-', '+-').replace(';+', ';').split('+'):
            if token[0] == '-':
                sign = -1
                token = token[1:]
            else:
                sign = 1

            if '*' in token:
                fac, key = token.split('*')
                if fac[0].isalpha():
                    fac, key = key, fac
                fac = sign * float(fac)
            else:
                fac, key = sign, token
            if key.lower() == 'faxc':
                fac_faxc += fac
            else:
                dftxc += '+'+str(fac)+'*'+key
        return fac_faxc, dftxc[1:]

    if ',' in description:
        x_code, c_code = description.split(',')
        fx,dft_x = _parse_guide(x_code)
        fc,dft_c = _parse_guide(c_code)
        fac_faxc = fx + fc
        dft_xc = dft_x + ',' + dft_c
    else:
        fac_faxc, dft_xc = _parse_guide(description)
    return fac_faxc, dft_xc

#controller
radi_method = dft.radi.gauss_chebyshev

def eval_vh(mol, coords, dm, Lvl=3, ang_lv=2): #only atom dependent values
    """Summary: Calculate real-space Hartree potential from given density matrix. See [Mirko2014]_ for some extra context.

        Args:
            mol (object) : an instance of :class:`Mole`
            coords (ndarray) :  grids space used for calculating Hartree potential
            dm (ndarray) : one-particle reduced density matrix in basis set representation
            Lvl (integer) : Interpolation grids space level (input : 0 ~ 9, default 3)
            ang_lv (integer) : setting a limit of angular momentum of spherical harmonics (input : 0 ~ 4, default 2)

        Returns:
            (ndarray): **vh** Pointwise Hartree potential
    """
    def _Cart_Spharm(xyz, lmax):
        """Summary: Cartesian spherical harmonics Z for given xyz coordinate

        Args:
            xyz (ndarray) : 3D coordinates to calculate spherical harmonics
            lmax (integer) : Maximum angular momentum quantum number to calculate spherical harmonics

        Returns:
            (array): from m = -l to l, return array Z of shape (ncoord, lmax+1, 2*lmax+1)
                for specific l & m, call Z[:, l, lmax+m]
        """
        ncoord = np.size(xyz[:, 0])
        rho = np.zeros((ncoord)) #distance from origin
        azi = np.zeros((ncoord)) #azimuth angle, theta in scipy
        pol = np.zeros((ncoord)) #polar angle, phi in scipy
        Z = np.zeros((ncoord, (lmax+1)**2))   #Real spherical harmonics

        xy = xyz[:, 0]**2 + xyz[:, 1]**2
        rho = np.sqrt(xy + xyz[:, 2]**2)
        azi = np.arctan2(xyz[:, 1], xyz[:, 0])
        pol = np.arctan2(np.sqrt(xy), xyz[:, 2])

        a = np.sqrt(0.5)
        for l in range(lmax+1):
            for m in range(1, l+1):
                Yp = sph_harm(m, l, azi, pol)
                Yn = sph_harm(-m, l, azi, pol)
                Z[:, l*(l+1)-m] = a*np.real(1.j*(Yn-((-1.)**m)*Yp))
                Z[:, l*(l+1)+m] = a*np.real((Yn+((-1.)**m)*Yp))

            Z[:, l*(l+1)] = np.real(sph_harm(0, l, azi, pol))

        return Z

    def _grid_refine(n_rad, n_ang, grid):
        """Summary: Reorder grids generated by PySCF for easy handling

        Args:
            n_rad (integer) : the number of radial grid
            n_ang (integer) : the number of angular grid
            grid (ndarray) : 1D sliced grid info (x, y, z coordinates or weights) generated by PySCF

        Returns:
            (ndarray): **Reordered gridpoints** n_ang grids belonging to the same radial grid
        """
        nrest = n_rad%12
        nmain = int((n_rad-nrest)/12)

        m = grid[:nmain*12*n_ang]
        m = m.reshape(nmain, n_ang, 12)
        mmain = np.zeros((nmain*12, n_ang))
        for i in range(nmain):
            mmain[i*12:(i+1)*12, :] = m[i, :, :].T

        m = grid[nmain*12*n_ang:]
        mrest = m.reshape(n_ang, nrest).T
        return np.concatenate((mmain, mrest), axis=0).reshape(n_rad*n_ang)

    def _eval_nang(lmax, lv=0):
        """Summary: Lebedev order based on maximum angular momentum of given basis set
        """
        LEBEDEV_ORDER_IDX = np.array([0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41,
                                      47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131])
        if lv == 0:   #default
            lgrids = 2*lmax+1
        elif lv == 1: #good
            lgrids = 2*lmax+7
        elif lv == 2: #verygood
            lgrids = 2*lmax+13
        elif lv == 3: #'excellent'
            lgrids = 2*lmax+20
        elif lv == 4: #'benchmark'
            lgrids = min(2*lmax+26, 131)
        minarg = np.argmin(abs(lgrids - LEBEDEV_ORDER_IDX))
        n_ang = dft.gen_grid.LEBEDEV_ORDER[LEBEDEV_ORDER_IDX[minarg]]
        return n_ang

    def eval_vhm(C, ZvH, d_atom, rad, lmax, n_coords, n_rad):
        """Summary: Python version of eval_vhm
        """

        def _eval_I1(C, rad, l, n_rad):
            I1 = np.zeros(n_rad)
            integrand = np.zeros((n_rad-1,4))
            tmp = np.zeros(n_rad-1)

            for j in range(4):
                integrand[:,j] = ( rad[1:]**(6.+l-j) - rad[:n_rad-1]**(6.+l-j) ) / (6.+l-j)

            for i in range(n_rad-2):
                tmp[i] = np.dot(integrand[i,:], C[:,i])
        
            for i in range(1,n_rad):
                I1[i] = I1[i-1] + tmp[i-1]
        
            return I1

        def _eval_I2(C, rad, l, n_rad):
            I2 =  np.zeros(n_rad)
            integrand = np.zeros((n_rad-1,4))
            tmp = np.zeros(n_rad-1)
            l = int(l)

            for j in range(4):
                if (5.-l-j) != 0 :
                    integrand[:,j] = ( rad[1:]**(5.-l-j) - rad[:n_rad-1]**(5.-l-j) ) / (5.-l-j)
                else:
                    integrand[:,6-l-1] = np.log(rad[1:]/rad[:n_rad-1])
        
            for i in range(n_rad-1):
                tmp[i] = np.dot(integrand[i,:], C[:,i])
        
            for i in range(1,n_rad):
                I2[n_rad-i-1] = I2[n_rad-i] + tmp[n_rad-i-1]

            return I2

        def _convert_coeff(c,x):
            nc=np.zeros_like(c)
            nx=np.size(x)
            for i in range(nx-1):
                nc[0,:]=c[0,:]
                nc[1,:]=c[1,:]-3*c[0,:]*x[:nx-1]
                nc[2,:]=c[2,:]-2*c[1,:]*x[:nx-1]+3*c[0,:]*x[:nx-1]**2
                nc[3,:]=c[3,:]-1*c[2,:]*x[:nx-1]+1*c[1,:]*x[:nx-1]**2-1*c[0,:]*x[:nx-1]**3
            return nc

        vh = np.zeros((ZvH.shape[0]))

        l = np.zeros((lmax+1)**2)
        for i in range(lmax+1):
            for j in range(-i, i):
                l[i*(i+1)+j] = i

        for i in range((lmax+1)**2):
            f=CubicSpline(rad,C[:,i])
            c=f.c
            c=_convert_coeff(c,rad)
            I1 = _eval_I1(c, rad, l[i], n_rad)
            I2 = _eval_I2(c, rad, l[i], n_rad)
            I = I1/(rad**(l[i]+1.)) + I2*(rad**(l[i]))
            f = CubicSpline(rad,I)
            v=f(d_atom)
            vh += ZvH[:,i]*v/(2.*l[i]+1.)
        return 4*np.pi*vh

    l_basis = np.max((mol._bas[:, 1]))
    l_margin = np.array([4, 6, 8, 12, 16])
    lmax = max(2*l_basis, l_basis + l_margin[ang_lv])

    Acoord = mol.atom_coords()
    d_atom = distance_matrix(Acoord, coords)
    symb = []
    n_rad = []
    n_ang = []
    for ia in range(mol.natm):
        symb.append(mol.atom_symbol(ia))
        chg = gto.charge(symb[ia])
        n_rad.append(dft.gen_grid._default_rad(chg, Lvl))
        n_ang.append(_eval_nang(lmax, lv=ang_lv))

    n_rad = np.max(n_rad)
    n_ang = np.max(n_ang)
    back1 = dft.gen_grid._default_rad
    back2 = dft.gen_grid._default_ang
    dft.gen_grid._default_rad = lambda *args: np.max(n_rad)
    dft.gen_grid._default_ang = lambda *args: np.max(n_ang)
    grids = dft.gen_grid.gen_atomic_grids(mol, radi_method=radi_method, prune=None, level=Lvl)
    dft.gen_grid._default_rad = back1
    dft.gen_grid._default_ang = back2

    rad, dr = radi_method(n_rad, chg)
    wrad = rad**2.*dr  #Radial weights
    sample_r = int(n_rad/2)
    ng = (grids[symb[0]][1]).size
    c = np.zeros((mol.natm, ng, 3))
    r = np.zeros((mol.natm, ng))
    w = np.zeros((mol.natm, ng))
    wang = np.zeros((mol.natm, n_ang))
    ao = np.zeros((mol.natm, ng, mol.nao_nr()), order='F')
    p = np.zeros((mol.natm, mol.natm, ng))
    ZI = np.zeros((mol.natm, n_ang, (lmax+1)**2), order='F')
    ZvH = np.zeros((mol.natm, coords.shape[0], (lmax+1)**2))

    #Density independent values
    for j, ia in enumerate(symb):    #j : idx , ia : name
        ca = np.array(grids[symb[j]][0]) #xyz coordinate centred at symb atom
        cx = _grid_refine(n_rad, n_ang, ca[:, 0])
        cy = _grid_refine(n_rad, n_ang, ca[:, 1])
        cz = _grid_refine(n_rad, n_ang, ca[:, 2])
        c[j] = np.vstack((cx, cy, cz)).T
        r[j] = np.linalg.norm(c[j], axis=1)
        ZI2 = _Cart_Spharm(c[j][sample_r*n_ang:(sample_r+1)*n_ang], lmax)
        wa = np.array(grids[symb[j]][1]) #weights
        w[j] = _grid_refine(n_rad, n_ang, wa)
        wang = w[j][sample_r*n_ang:(sample_r+1)*n_ang]/wrad[sample_r] #Angular weights
        ZI[j] = np.einsum('al,a->al', ZI2, wang)
        tst = c[j]+Acoord[j]
        d = distance_matrix(Acoord, tst) #the difference between newly define grids and original grids
        rel_coord = coords-Acoord[j, :]
        """.. todo::
            * Multi threading on "ZvH[j]=_Cart_Spharm(rel_coord, lmax)"
        """
        ZvH[j] = _Cart_Spharm(rel_coord, lmax) #This is time consuming
        #partition function P_i
        p[j] = np.exp(-2.*d)/(d**3) #partition function P_i
        for ia, za in enumerate(mol.atom_charges()):
            if za==1: #Special treatment on hydrogen atom
                p[j, ia, :] *= 0.3
        ao[j] = numint.eval_ao(mol, tst) #AO value in real coordinate

    #Density dependent values
    vH = np.zeros(int(coords.size/3))
    for i in range(mol.natm):
        idx = np.argsort(d_atom[i])
        rho_org = numint.eval_rho(mol, ao[i], dm) #Rho in real coordinate
        rho_here = p[i, i, :] / np.sum(p[i], axis=0)*rho_org #Eq(4)
        rho_here = rho_here.reshape(n_rad, n_ang) #(r,\theta \phi)
        #r : n_rad, a : n_ang, l : (lmax+1)**2
        C = np.matmul(rho_here, ZI[i])
        if kf_imported:
            vH[idx] += kspies_fort.eval_vhm(C, ZvH[i, idx, :], d_atom[i, idx], rad, lmax, coords.shape[0], n_rad)
        else:
            vH[idx] += eval_vhm(C, ZvH[i, idx, :], d_atom[i, idx], rad, lmax, coords.shape[0], n_rad)

    return vH

def eval_vxc(mol, dm, xc_code, coords, delta=1e-7):
    """Summary: Calculate real-space exchange-correlation potential for GGA from given density matrix

        Args:
            mol (object) : an instance of :class:`Mole`
            coords (ndarray) : grids space used for calculating XC potential
            dm (ndarray) : one-particle reduced density matrix in basis set representation
            xc_code (str) : XC functional description.
            delta (float) : amount of finite difference to calculate numerical differentiation, default is 1e-7 a.u.

        Returns:
            (tuple): tuple containing:

                (ndarray) Pointwise XC potential, vxc(r) for RKS dm, vxc_alpha(r) for UKS dm

                (ndarray) Pointwise XC potential, vxc(r) for RKS dm, vxc_beta(r) for UKS dm


    """
    Ncoord = np.size(coords)//3
    ao = numint.eval_ao(mol, coords, deriv=1)

    def _spin_tr(dm):
        """Summary: Return spin-traced density matrix
        """
        if np.array(dm).ndim == 3:
            return dm[0]+dm[1]
        return dm

    def _numderiv(aux, delta):
        """Summary: numerical differentiation of 3D function
        """
        nabla_res = np.zeros((3, Ncoord))
        nabla_res[0, :] = (aux[0, :]-aux[1, :])/(2.*delta)
        nabla_res[1, :] = (aux[2, :]-aux[3, :])/(2.*delta)
        nabla_res[2, :] = (aux[4, :]-aux[5, :])/(2.*delta)
        return nabla_res

    auxcoords = np.zeros((6, Ncoord, 3))
    for i in range(6):
        auxcoords[i, :, :] = coords[:, :]
        auxcoords[i, :, (i//2)] = coords[:, (i//2)]+delta*(-1.)**i

    if mol.spin == 0: #spin-unpolarized case
        dm = _spin_tr(dm)

        den = numint.eval_rho(mol, ao, dm, xctype='GGA')
        exc, vxc = dft.libxc.eval_xc(xc_code, den, spin=mol.spin, deriv=1)[:2]

        auxvsigma = np.zeros((6, Ncoord))
        for i in range(6):
            auxao = numint.eval_ao(mol, auxcoords[i, :, :], deriv=1)
            auxden = numint.eval_rho(mol, auxao, dm, xctype='GGA')
            vxc = dft.libxc.eval_xc(xc_code, auxden, spin=mol.spin, deriv=1)[1]
            auxvsigma[i, :] = vxc[1]

        ao = numint.eval_ao(mol, coords, deriv=2)
        den = numint.eval_rho(mol, ao, dm, xctype='mGGA')
        nabla_vsigma = _numderiv(auxvsigma, delta)
        vxc = vxc[0]-2*(den[4, :]*vxc[1]+np.einsum('ir,ir->r', den[1:4, :], nabla_vsigma[:, :]))
        if np.array(dm).ndim == 2: #RKS scheme
            return np.array(vxc)
        elif np.array(dm).ndim == 3: #UKS scheme
            return np.array(vxc), np.array(vxc)

    elif mol.spin != 0: #spin-polarized case
        den_a = numint.eval_rho(mol, ao, dm[0], xctype='GGA')
        den_b = numint.eval_rho(mol, ao, dm[1], xctype='GGA')
        exc, vxc = dft.libxc.eval_xc(xc_code, (den_a, den_b), spin=mol.spin, deriv=1)[:2]

        auxvsigma_aa = np.zeros((6, Ncoord))
        auxvsigma_ab = np.zeros((6, Ncoord))
        auxvsigma_bb = np.zeros((6, Ncoord))
        for i in range(6):
            auxao = numint.eval_ao(mol, auxcoords[i, :, :], deriv=1)
            auxden_a = numint.eval_rho(mol, auxao, dm[0], xctype='GGA')
            auxden_b = numint.eval_rho(mol, auxao, dm[1], xctype='GGA')
            vxc = dft.libxc.eval_xc(xc_code, (auxden_a, auxden_b), spin=mol.spin, deriv=1)[1]
            auxvsigma_aa[i, :] = vxc[1][:, 0]
            auxvsigma_ab[i, :] = vxc[1][:, 1]
            auxvsigma_bb[i, :] = vxc[1][:, 2]

        nabla_vsigma_aa = _numderiv(auxvsigma_aa, delta)
        nabla_vsigma_ab = _numderiv(auxvsigma_ab, delta)
        nabla_vsigma_bb = _numderiv(auxvsigma_bb, delta)

        ao = numint.eval_ao(mol, coords, deriv=2)
        den_a = numint.eval_rho(mol, ao, dm[0], xctype='mGGA')
        den_b = numint.eval_rho(mol, ao, dm[1], xctype='mGGA')

        vxc_a = vxc[0][:, 0]\
        -2.*(den_a[4, :]*vxc[1][:, 0]+np.einsum('ir,ir->r', den_a[1:4, :], nabla_vsigma_aa[:, :]))\
        -np.einsum('ir,ir->r', nabla_vsigma_ab, den_b[1:4, :])-den_b[4, :]*vxc[1][:, 1]

        vxc_b = vxc[0][:, 1]\
        -2.*(den_b[4, :]*vxc[1][:, 2]+np.einsum('ir,ir->r', den_b[1:4, :], nabla_vsigma_bb[:, :]))\
        -np.einsum('ir,ir->r', nabla_vsigma_ab, den_a[1:4, :])-den_a[4, :]*vxc[1][:, 1]

        return np.array(vxc_a), np.array(vxc_b)

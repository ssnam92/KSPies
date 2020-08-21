from pyscf import gto, scf, dft 
import numpy as np
import wy_dev as wy
import zmp_dev as zmp
import util_dev as util
"""Blackbox test script"""

#Reference simple HF run
mol = gto.M(atom = 'Be',
            basis = 'cc-pVTZ',
            spin = 0,
            verbose = 0 )
mf = scf.RHF(mol).run()
dm_rhf = mf.make_rdm1()
mf = scf.UHF(mol).run()
dm_uhf = mf.make_rdm1()

print("Check WY")
print("!Expected -Ws = -14.57272496 !")

mw1 = wy.RWY(mol, dm_rhf)
mw1.reg = 1e-5
mw1.run()
mw1.info()

mw2 = wy.UWY(mol, dm_uhf)
mw2.reg = 1e-5
mw2.run()
mw2.info()

print()

print("Check ZMP")
print("!Expected :                 gap=  0.1262392 dN=   78.65 C= 6.75e-04 !")

mz1 = zmp.RZMP(mol, dm_rhf)
mz1.zscf(16)

mz2 = zmp.UZMP(mol, dm_uhf)
mz2.zscf(16)

print()

#Check util giving correct answer
grids = dft.gen_grid.Grids(mol)
grids.build()
coords = grids.coords
weights = grids.weights
ao = dft.numint.eval_ao(mol, coords, deriv=1)
rho = dft.numint.eval_rho(mol, ao, dm_rhf, xctype='GGA')

vj_ana = scf.hf.get_jk(mol, dm_rhf)[0]
Eh_ana = .5*np.einsum('ij,ji', vj_ana, dm_rhf)

print("Check util.eval_vh")
print("Eh(numerical) - Eh(analytic) in Hartrees")
saved = [-0.000269882, -0.000132191, -0.000072399, 
         -0.000042809, -0.000026879, -0.000017730] #Precomputed values
print("Lvl  computed      saved")
for i, Lvl in enumerate([ 3, 4, 5, 6, 7, 8]):
  vj_num = util.eval_vh(mol, coords, dm_rhf, Lvl = Lvl)
  Eh_num = .5*np.einsum('r,r,r', vj_num, rho[0,:], weights)
  print('%i  %.7f %.7f'%(Lvl,(Eh_num-Eh_ana),saved[i]))

print()
print("Check util.eval_vxc_libxc")
print("Check with vxc*n differences on same density")

xc_code = 'pbe'
_, vxc, _, _ = dft.libxc.eval_xc(xc_code, rho)
Vxc = dft.numint.eval_mat(mol, ao, weights, rho, vxc, xctype='GGA' ) #PySCF Vxc matrix
Exc = np.einsum('ij,ji', Vxc, dm_rhf) #analytic Vxc*n 
vxcr = util.eval_vxc_libxc(mol, dm_rhf, xc_code, coords) 
Exc_num = np.einsum('r,r,r', vxcr, rho[0,:], weights) #numerical Vxc*n
print('PySCF - util  : %.7f'%(Exc-Exc_num))

xc_code2 = '0.999*pbe,pbe'
_, vxc, _, _ = dft.libxc.eval_xc(xc_code2, rho)
Vxc2 = dft.numint.eval_mat(mol, ao, weights, rho, vxc, xctype='GGA' )
Exc2 = np.einsum('ij,ji', Vxc2, dm_rhf)
print('PBE - 0.999PBE: %.7f'%(Exc-Exc2))


import numpy as np
from pyscf import gto, scf, lib, dft, cc
from kspies import zmp, wy, util
lib.num_threads(8)

#Define system
mol=gto.M(atom= '''
O 1.208 0.0 0.0
O 0.0   0.0 0.0
''',
basis = 'cc-pVQZ',spin=2,  verbose=3)

#Reference calculation
#mf=scf.UHF(mol)
#mf.kernel()
#mc=cc.CCSD(mf)
#mc.kernel()
#dm_tar = util.mo2ao(mf.mo_coeff, mc.make_rdm1())
#np.save('P_tar',dm_tar)

dm_tar=np.load('P_tar.npy')

#Grid generation for density accuracy check
grids=dft.gen_grid.Grids(mol)
grids.build()
coords=grids.coords
weights=grids.weights
ao=dft.numint.eval_ao(mol, coords)
def ndiff(P):
  if P.ndim==3: P=P[0]+P[1]
  rho=dft.numint.eval_rho(mol, ao, P)
  return np.einsum('r,r',abs(rho),weights)

#DF-UZMP
mz =zmp.UZMP(mol, dm_tar)
mz.with_df = True
for l in [ 8, 16, 32, 64, 128, 256, 512 ]:
    mz.level_shift = l*0.1
    mz.zscf(l)

#UWY
mw = wy.UWY(mol, dm_tar)
mw.run()
mw.info()
dm_wy = mw.dm
print("RWY density difference: ",ndiff(dm_wy-dm_tar))


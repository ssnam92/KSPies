from pyscf import gto,scf,lib,dft
import numpy as np
import zmp_dev as zmp
lib.num_threads(8)

#Define system
mol=gto.M(atom= '''
C    0.0000    1.3936     0.0000
C    1.2069    0.6968     0.0000
C    1.2069   -0.6968     0.0000
C    0.0000   -1.3936     0.0000
C   -1.2069   -0.6968     0.0000
C   -1.2069    0.6968     0.0000
H    0.0000    2.4787     0.0000
H    2.1467    1.2394     0.0000
H    2.1467   -1.2394     0.0000
H    0.0000   -2.4787     0.0000
H   -2.1467   -1.2394     0.0000
H   -2.1467    1.2394     0.0000
''',
basis = 'cc-pVQZ', verbose=4)

#Reference HF calculation
mf=scf.RHF(mol).density_fit()
mf.kernel()
P_tar = mf.make_rdm1()

import wy_dev as wy
myWY=wy.RWY(mol,P_tar)
myWY.run()
myWY.info()
myWY.time_profile()
dm=myWY.dm

grids=dft.gen_grid.Grids(mol)
grids.build()
coords=grids.coords
weights=grids.weights
ao=dft.numint.eval_ao(mol,coords)
rho=dft.numint.eval_rho(mol,ao,P_tar-dm)
print(np.einsum('r,r',abs(rho),weights))

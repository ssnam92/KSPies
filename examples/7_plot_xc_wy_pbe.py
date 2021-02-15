import numpy as np
from pyscf import gto, scf, dft
from kspies import wy, util
import matplotlib.pyplot as plt

mol = gto.M(atom='Ne',
            basis='aug-cc-pVTZ')
mf=scf.RHF(mol).run()
dm_tar = mf.make_rdm1()

#Define plotting domain
coords = []
for x in np.linspace(0, 3, 1001):
    coords.append((x, 0., 0.))
coords = np.array(coords)

plt.figure(figsize=(3,4))

#WY calculations
mw = wy.RWY(mol, dm_tar, pbas='cc-pVQZ')
mw.guide = 'pbe'
mw.tol = 1e-7
pb = dft.numint.eval_ao(mw.pmol, coords) #potential basis values on grid
vg = util.eval_vxc(mol, dm_tar, mw.guide, coords) #guiding potential on grid

for expo in np.arange(2,6):
    mw.reg = 10.**(-expo)
    mw.run()
    vC = np.einsum('t,rt->r', mw.b, pb)
    mw.info()
    plt.plot(coords[:,0], vg+vC, label=r'$\eta$ = 10^'+str(-expo))

plt.xlabel("x")
plt.ylabel("vx(r)")
plt.xlim(0, 3)
plt.ylim(-5, 0)
plt.legend()
plt.subplots_adjust(left=0.2, right=0.93, bottom=0.12, top=0.97)
plt.show()

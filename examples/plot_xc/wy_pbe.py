from pyscf import gto, scf, dft
import matplotlib.pyplot as plt
import numpy as np
import util_dev as util
import wy_dev as wy

mol = gto.M(atom='Ne',
            basis='aug-cc-pVTZ')
mf=scf.RHF(mol).run()
dm_tar = mf.make_rdm1()

#Define plotting domain
coords = []
for x in np.linspace(0, 5, 1001):
    coords.append((x, 0., 0.))
coords = np.array(coords)

#WY calculations
mw = wy.RWY(mol, dm_tar, pbas='cc-pVQZ')
mw.guide = 'pbe'
mw.tol = 1e-7
pb = dft.numint.eval_ao(mw.mol2, coords) #potential basis values on grid
vg = util.eval_vxc_libxc(mol, dm_tar, mw.guide, coords) #guiding potential on grid

for expo in np.arange(2,6):
  mw.reg = 10.**(-expo)
  mw.run()
  vC = np.einsum('t,rt->r', mw.b, pb)
  mw.time_profile()
  plt.plot(coords[:,0], vg+vC, label=r'$\alpha$ = 10^'+str(-expo))

plt.xlim(0, 5)
plt.ylim(-9, 0)
plt.legend()
plt.savefig('vxc_wy_pbe.pdf', format='pdf')
plt.savefig('vxc_wy_pbe.eps', format='eps')
plt.show()

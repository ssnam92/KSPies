import numpy as np
from pyscf import gto, scf
from kspies import zmp, util
import matplotlib.pyplot as plt

#Target density from HF
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

#ZMP calculations
mz = zmp.RZMP(mol, dm_tar)
mz.guide='faxc'
for l in [ 16, 128, 1024 ]:
    mz.level_shift = 0.1*l
    mz.zscf(l) 
    dmxc = l*mz.dm-(l+1./mol.nelectron)*dm_tar
    vxc = util.eval_vh(mol, coords, dmxc )
    plt.plot(coords[:, 0], vxc, label = r'$\lambda$='+str(l))

plt.xlabel("x")
plt.ylabel("vx(r)")
plt.xlim(0, 3)
plt.ylim(-5, 0)
plt.legend()
plt.subplots_adjust(left=0.2, right=0.93, bottom=0.12, top=0.97)
plt.show()

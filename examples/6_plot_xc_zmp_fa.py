from pyscf import gto, scf
import matplotlib.pyplot as plt
import numpy as np
import util
import zmp

#Target density from HF
mol = gto.M(atom='Ne',
            basis='aug-cc-pVTZ')
mf=scf.RHF(mol).run()
dm_tar = mf.make_rdm1()

#Define plotting domain
coords = []
for x in np.linspace(0, 5, 1001):
    coords.append((x, 0., 0.))
coords = np.array(coords)

#ZMP calculations
mz = zmp.RZMP(mol, dm_tar)
mz.guide='faxc'
for l in [ 16, 128, 1024 ]:
    mz.level_shift = 0.1*l
    mz.zscf(l) 
    dmxc = l*mz.dm-(l+1./mol.nelectron)*dm_tar
    vxc = util.eval_vh(mol, coords, dmxc )
    plt.plot(coords[:, 0], vxc, label = r'$\lambda$='+str(l))

plt.xlim(0, 5)
plt.ylim(-9, 0)
plt.legend()
#plt.savefig('vxc_zmp_fa.pdf', format='pdf')
#plt.savefig('vxc_zmp_fa.eps', format='eps')
plt.show()

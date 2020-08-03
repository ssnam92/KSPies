from pyscf import gto, scf
import numpy as np
import wy_dev as wy

mol = gto.M(atom = 'N 0 0 0 ; N 1.1 0 0', 
            basis = 'cc-pVDZ')
mf = scf.RHF(mol).run()
dm_tar = mf.make_rdm1()

PBS = gto.expand_etbs([(0, 13, 2**-4 , 2),
                       (1, 3 , 2**-2 , 2)])
mw=wy.RWY(mol, dm_tar, pbas=PBS)
mw.tol = 2e-7
alphas = [ 2.**(-a) for a in np.linspace(5., 27., 45) ]
v = np.zeros(len(alphas))
W = np.zeros(len(alphas))
for i, alpha in enumerate(alphas):
    mw.reg=alpha
    mw.run()
    v[i] = mw.Dvb()
    W[i] = mw.Ws

mw.reg=0.
mw.run()
Ws_fin = mw.Ws

import matplotlib.pyplot as plt
fig,ax = plt.subplots(2)
ax[0].scatter(np.log10(Ws_fin-W), np.log10(v))
ax[1].scatter(np.log10(alphas), v*alphas/(Ws_fin-W))

plt.tight_layout()
plt.savefig('L_curves.pdf', format='pdf')
plt.savefig('L_curves.eps', format='eps')
plt.show()

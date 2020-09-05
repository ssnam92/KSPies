from pyscf import gto, scf
import numpy as np
import wy

mol = gto.M(atom = 'N 0 0 0 ; N 1.1 0 0', 
            basis = 'cc-pVDZ')
mf = scf.RHF(mol).run()
dm_tar = mf.make_rdm1()

PBS = gto.expand_etbs([(0, 13, 2**-4 , 2),
                       (1, 3 , 2**-2 , 2)])
mw = wy.RWY(mol, dm_tar, pbas=PBS)
#Note that for this designed-to-be ill-conditioned problem,
#Hessian-based optimization algorithms are problematic.
mw.method = 'bfgs'
mw.tol = 2e-7
mw.run()
mw.info()
Ws_fin = mw.Ws

etas = [ 2.**(-a) for a in np.linspace(5., 27., 45) ]
v = np.zeros(len(etas))
W = np.zeros(len(etas))
for i, eta in enumerate(etas):
    mw.reg=eta
    mw.run()
    v[i] = mw.Dvb()
    W[i] = mw.Ws
    mw.info()

import matplotlib.pyplot as plt
fig,ax = plt.subplots(2)
ax[0].scatter(np.log10(Ws_fin-W), np.log10(v))
ax[1].scatter(np.log10(etas), v*etas/(Ws_fin-W))

plt.tight_layout()
#plt.savefig('L_curves.pdf', format='pdf')
#plt.savefig('L_curves.eps', format='eps')
plt.show()

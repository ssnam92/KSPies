from pyscf import lib,dft,gto,scf,ao2mo,cc
import wy_dev as wy
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.linalg import eigh

lib.num_threads(8)

x=np.linspace(-10,10,201)
h=(x[-1]-x[0])/(len(x)-1)
n=len(x)

a=np.zeros(len(x))
a[0]=2
a[1]=-1
T=toeplitz(a,a)/(2*h**3)

S=np.identity(len(x))/h
k=0.25
V=np.diag(0.5*k*x**2)/h

l=0.5
def deno(l):
  b=np.expand_dims(x,axis=0)
  c=b.T
  dist=abs(b-c)
  return 1./np.sqrt(dist**2+l**2)

def get_J(P):
  J=np.diag(np.einsum('ii,ik->k',P,deno(l)))/(h**2)
  return J

def get_K(P):
  K=np.einsum('il,il->il',P,deno(l))/(h**2)
  return K

mol=gto.M()
n=len(x)
mol.nelectron=4
mol.verbose=0
mol.incore_anyway=True

F=T+V
for i in range(30):
  e,C=eigh(F,S)
  P=2*np.einsum('ik,jk->ij',C[:,:mol.nelectron//2],C[:,:mol.nelectron//2])
  J=get_J(P)
  K=get_K(P)
  F=T+V+J-0.5*K
  print("EHF = ",np.einsum('ij,ji',T+V+0.5*J-0.25*K,P))
P_tar=P
plt.plot(x,10*np.diag(P_tar)/h**2,label='den(HF)',color='black')

Sijt=np.zeros((n,n,n))
for i in range(n):
  Sijt[i,i,i]=1./h**2

mw = wy.RWY(mol,P_tar,Sijt=Sijt)
mw.tol=1e-8
#mw.method='bfgs'
mw.T = T
mw.Tp = T
mw.V = V
mw.S = S
mw.guide = None
mw.run()
mw.info()
mw.time_profile()
Vb=np.diag(mw.b-mw.b[50])/(h**2)
plt.plot(x,10*np.diag(mw.dm)/h**2,label='den(WY)',color='red')
plt.plot(x,np.diag(V)*h,label=r'$v_{ext}$(r)')
plt.plot(x,np.diag(V+Vb)*h,label=r'$v_{S}$(r)')
plt.xlim(-10,10)
plt.ylim(0,10)
plt.tight_layout()

#F=T+V+Vb
#e,C=eigh(F,S)
#P=2*np.einsum('ik,jk->ij',C[:,:mol.nelectron//2],C[:,:mol.nelectron//2])
#plt.plot(x,np.diag(P)/h,label='den(recon)',color='blue')

plt.legend()
plt.show()
#plt.savefig('Harmonic.pdf', format='pdf')
#plt.savefig('Harmonic.eps', format='eps')



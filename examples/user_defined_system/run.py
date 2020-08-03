from pyscf import lib
import wy_dev as wy
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.linalg import eigh

lib.num_threads(8)
#Define system
x = np.linspace(-10, 10, 201) #Domain
h = (x[-1]-x[0])/(len(x)-1) #grid spacing
n = len(x)                  # # of grid = # of basis

a=np.zeros(len(x))
a[0]=2
a[1]=-1
T=toeplitz(a,a)/(2*h**3)  #Kinetic energy matrix by 2nd order FD

S=np.identity(len(x))/h   #Overlap matrix
k=0.25
V=np.diag(0.5*k*x**2)/h   #Harmonic potential matrix

l=0.5                     #1D e-e soft repulsion parameter
def deno(l):
  b=np.expand_dims(x,axis=0)
  c=b.T
  dist=abs(b-c)
  return 1./np.sqrt(dist**2+l**2)

def get_J(dm):
  J=np.diag(np.einsum('ii,ik->k', dm, deno(l)))/(h**2)
  return J

def get_K(dm):
  K=np.einsum('il,il->il', dm, deno(l))/(h**2)
  return K

#Pass to mole object
mol=gto.M()
mol.nelectron=4
mol.verbose=0
mol.incore_anyway=True

#Solve HF equation
F=T+V
for i in range(30):
  e,C=eigh(F,S)
  dm=2*np.einsum('ik,jk->ij',C[:,:mol.nelectron//2],C[:,:mol.nelectron//2])
  J=get_J(dm)
  K=get_K(dm)
  F=T+V+J-0.5*K
  print("EHF = ",np.einsum('ij,ji',T+V+0.5*J-0.25*K,dm))
dm_tar=dm
plt.plot(x,10*np.diag(dm_tar)/h**2,label='den(HF)',color='black') # x10 scaled density

#Three-center overlap matrix, each diagonal= (1/h)^3 *h 
Sijt=np.zeros((n,n,n))
for i in range(n):
  Sijt[i,i,i]=1./h**2

#Run WY
mw = wy.RWY(mol,dm_tar,Sijt=Sijt)
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

#Plotting
Vb=np.diag(mw.b-mw.b[50])/(h**2) #KS potential is unique up to a constant. 
plt.plot(x,10*np.diag(mw.dm)/h**2,label='den(WY)',color='red') # x10 scaled density
plt.plot(x,np.diag(V)*h,label=r'$v_{ext}$(r)')
plt.plot(x,np.diag(V+Vb)*h,label=r'$v_{S}$(r)')
plt.xlim(-10,10)
plt.ylim(0,10)
plt.tight_layout()

#Reconstruct Fock matrix from inversion result and re-evaluate density
#F=T+V+Vb
#e,C=eigh(F,S)
#dm=2*np.einsum('ik,jk->ij',C[:,:mol.nelectron//2],C[:,:mol.nelectron//2])
#plt.plot(x,np.diag(dm)/h,label='den(recon)',color='blue')

plt.legend()
plt.show()
#plt.savefig('Harmonic.pdf', format='pdf')
#plt.savefig('Harmonic.eps', format='eps')



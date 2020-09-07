import numpy as np
from pyscf import lib, gto
from kspies import wy
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from scipy.linalg import eigh

#Define system
x = np.linspace(-10, 10, 201) #Domain
h = (x[-1]-x[0])/(len(x)-1)   #grid spacing
n = len(x)                    #Dimension of basis
a = np.zeros(len(x))
a[0] = 2.
a[1] = -1.
T = toeplitz(a,a)/(2*h**2)  #Kinetic energy matrix by 2nd order FD

S = np.identity(len(x))   #Overlap matrix
k = 0.25
V = np.diag(0.5*k*x**2)   #Harmonic potential matrix

l = 0.5                     #1D e-e soft repulsion parameter
def deno(l):
  b = np.expand_dims(x, axis=0)
  dist = abs(b-b.T)
  return 1./np.sqrt(dist**2+l**2)

def get_J(dm):
  J = np.diag(np.einsum('ii,ik->k', dm, deno(l)))
  return J

def get_K(dm):
  K=np.einsum('il,il->il', dm, deno(l))
  return K

#Pass to mole object
mol = gto.M()
mol.nelectron = 4
mol.verbose = 0
mol.incore_anyway = True

#Solve HF equation
F = T+V
for i in range(30):
  e,C = eigh(F,S)
  dm = 2*np.einsum('ik,jk->ij',C[:,:mol.nelectron//2],C[:,:mol.nelectron//2])
  J = get_J(dm)
  K = get_K(dm)
  F = T+V+J-0.5*K
  print("EHF = ",np.einsum('ij,ji', T+V+0.5*J-0.25*K, dm))
dm_tar = dm

plt.plot(x, 10*np.diag(dm_tar)/h, label='den(HF)', color='black') # x10 scaled density

#Three-center overlap integral
Sijt=np.zeros((n,n,n))
for i in range(n):
  Sijt[i,i,i]=1.

#Run WY
mw = wy.RWY(mol, dm_tar, Sijt=Sijt)
mw.tol = 1e-7
mw.method = 'bfgs'
mw.T = T
mw.Tp = T
mw.V = V
mw.S = S
mw.guide = None
mw.run()
mw.info()
mw.time_profile()

#Plotting
Vb = np.diag(mw.b) #-mw.b[50]) #KS potential is unique up to a constant. 
plt.plot(x, 10*np.diag(mw.dm)/h, label='den(WY)', color='red', linestyle='--') # x10 scaled density
plt.plot(x, np.diag(V), label=r'$v_{ext}$(r)')
plt.plot(x, np.diag(V+Vb), label=r'$v_{S}$(r)')
plt.plot(x, 1e+6*np.diag(mw.dm-dm_tar)/h,label='den(WY-HF)', color='blue', linestyle='--') # x10^6 scaled diff
plt.xlim(-10, 10)
plt.ylim(-0.5, 10)
plt.tight_layout()

plt.legend()
plt.show()


import numpy as np
from pyscf import gto, scf, dft 
from kspies import wy

mol_1 = gto.M(atom='Ne', basis='aug-cc-pVTZ')
mf = scf.UHF(mol_1).run()
P_tar_1 = mf.make_rdm1()

def make_sto(zeta):
    def sto(coords):
        dist = np.sum(coords**2, axis=1)**.5
        return np.exp(-zeta*dist)
    return sto

pbas = []
for zeta in [ 0.25, 0.5, 1., 2., 4., 8. ]:
    pbas.append(make_sto(zeta))

mol_3=gto.M(atom='Ne', basis='aug-cc-pVTZ')
def make_gto(i): #make aug-cc-pVTZ as user-defined basis
    def gto(coords):
        return dft.numint.eval_ao(mol_3, coords)[:,i]
    return gto

#If this pbas is used, result is same with setting
#pbas='aug-cc-pVTZ'
#pbas = []
#for i in range(mol_3.nao_nr()):
#  pbas.append(make_gto(i))

mywy = wy.UWY(mol_1, P_tar_1, pbas=pbas)
mywy.run()
mywy.info()

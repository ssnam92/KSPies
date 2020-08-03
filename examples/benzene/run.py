from pyscf import gto,scf,lib,dft
import numpy as np
import zmp_dev as zmp
lib.num_threads(8)

#Define system
mol=gto.M(atom= '''
C    0.0000    1.3936     0.0000
C    1.2069    0.6968     0.0000
C    1.2069   -0.6968     0.0000
C    0.0000   -1.3936     0.0000
C   -1.2069   -0.6968     0.0000
C   -1.2069    0.6968     0.0000
H    0.0000    2.4787     0.0000
H    2.1467    1.2394     0.0000
H    2.1467   -1.2394     0.0000
H    0.0000   -2.4787     0.0000
H   -2.1467   -1.2394     0.0000
H   -2.1467    1.2394     0.0000
''',
basis = 'cc-pVTZ', verbose=3)

#Reference HF calculation
mf=scf.RHF(mol).density_fit()
mf.kernel()
P_tar = mf.make_rdm1()

#run ZMP without density fit (slow)
#mz1 = zmp.RZMP(mol, P_tar)
#mz1.with_df = False
#for l in [ 8, 16, 32, 64, 128, 256, 512 ]:
#    mz1.level_shift_factor = l*0.1
#    mz1.zscf(l)
#    np.save('Pn'+str(l), mz1.dm)
  
#run ZMP with density fit (fast)
#mz2=zmp.RZMP(mol, P_tar)
#mz2.with_df = True
#mz2.diis_space = 20
#for l in [ 8, 16, 32, 64, 128, 256, 512 ]:
#    mz2.level_shift_factor = l*0.1
#    mz2.zscf(l)
#    np.save('Pf'+str(l), mz2.dm)

#Setting for plotting density difference
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
      [c('#E94B00'), c('#FF9021'), 0.30,
       c('#FF9021'), c('#FFFBEF'), 0.49,
       c('white')  , 0.51,
       c('#F4FFDD'), c('#7BFF0D'), 0.80,
       c('#7BFF0D'), c('#0AE000')])

#Define drawing region
unit = 1.88973
nx = 101 
ny = 101
xs = unit*np.linspace(-3, 3, nx)
ys = unit*np.linspace(-3, 3, ny)
iz = 0.
coords = []
for ix in xs:
    for iy in ys:
        coords.append((ix, iy, iz))
coords=np.array(coords)
ao=dft.numint.eval_ao(mol,coords)

ls = [8, 32, 128, 512]
fig = plt.figure(figsize = (8, 7))
ax = fig.subplots(2, 2, sharex=True, sharey=True, gridspec_kw={'hspace': 0.06, 'wspace': 0.06})
for i in range(4):
    P=np.load('Pf'+str(ls[i])+'.npy')
    drho=dft.numint.eval_rho(mol,ao,P-P_tar)
    drho=np.reshape(drho, (nx, ny))
    c = ax[i//2,i%2].contourf(xs/unit, ys/unit, drho.T, levels=np.linspace(-0.05,0.05,101), cmap=rvb, extend='both')
    ax[i//2,i%2].set_aspect('equal', adjustable='box')

plt.subplots_adjust(top=0.95,right=0.84,bottom=0.06,left=0.06)
cbar_ax = fig.add_axes([0.88, 0.06, 0.03, 0.88])
fig.colorbar(c, cax=cbar_ax,ticks=np.linspace(-0.05,0.05,11))
plt.savefig('bz_dendiff.pdf', format='pdf')
#plt.savefig('bz_dendiff.eps', format='eps')
plt.show()


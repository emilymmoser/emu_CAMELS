import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import scipy.interpolate
import numpy as np
import warnings
import random
import time
import ostrich.emulate
import ostrich.interpolate
import sklearn.gaussian_process.kernels as skgp_kernels
import functions as fs
import sys

time1=time.time()

home='/home/cemoser/Repositories/ostrich/Emulator_profiles/'
suite=sys.argv[1]
vary_str=sys.argv[2]
prof=sys.argv[3]

mass=fs.mass
mass_str=fs.mass_str
snap=fs.snap
z=fs.choose_redshift(suite)
vary,sims=fs.choose_vary(vary_str)
samples=fs.cartesian_prod(vary,z,mass)
nsamp=samples.shape[0]
usecols,ylabel=fs.choose_profile(prof)

x,y,errup,errlow,stddev=fs.load_profiles_3D(usecols,home,suite,sims,snap,mass_str,prof)
y=np.transpose(y)
errup=np.transpose(errup)
errlow=np.transpose(errlow)
stddev=np.transpose(stddev)
'''
def get_errs_rbf(samps,data,true_coord,true_data,function):
    emulator = ostrich.emulate.PcaEmulator.create_from_data(
        samps,
        data.reshape(data.shape[0],-1),
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function': function},
        num_components=12,
    )
    emulated = emulator(true_coord)
    emulated=emulated.reshape(len(data))
    err=((emulated - true_data)/true_data).squeeze()
    return emulated,err



#GaussianProcessInterpolator, Rbf, Polynomial
emulator = ostrich.emulate.PcaEmulator.create_from_data(
    samples,
    y,
    ostrich.interpolate.RbfInterpolator,
    interpolator_kwargs={'function': 'linear'},
    num_components=12,
)

errs_drop_1 = np.zeros_like(y)
emulated_drop_1=np.zeros_like(y)
start=time.time()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for i in range(nsamp):
        emulated_drop_1[:,i],errs_drop_1[:,i] = get_errs_rbf(
            np.delete(samples, i,0),
            np.delete(y,i,1),
            samples[i:i+1],
            y[:,i],'linear',
        )

'''
errs_drop_1=np.genfromtxt('./Error_histograms/errs_drop1_'+suite+'_'+vary_str+'_'+prof+'_4Mbins_RbfInterp_lin.txt')

def chunks(indices,n):
    return [indices[i:i+n] for i in range(0,len(indices),n)]

total_indices=np.linspace(0,nsamp-1,nsamp,dtype='int')
#print("total indices",len(total_indices))
chunked=chunks(total_indices,len(snap)*len(mass))
#print("split into chunks of size len(snap)",np.shape(chunked))


#for 2D, each of these should be 10 redshifts*13 radial points=130
#for 3d, each of these 10 redshifts*13 radial*4 masses=520?
mean_radial_errs=[]
for i in range(len(vary)):
    idx=chunked[i]
    new_errs=errs_drop_1[:,idx]
    new_errs=np.reshape(new_errs,-1)
    mean_radial_errs.append(np.mean(np.log10(new_errs[new_errs>0])))
#print(np.shape(mean_radial_errs))

fig=plt.figure()
ax=plt.gca()

ax.scatter(vary,mean_radial_errs)
ax.set_ylabel('log (percent error)')
ax.axhline(-1, label=r'$10\%$ error level', color='red')
ax.axhline(-2, label=r'$1\%$ error level', color='orange')
ax.axhline(-3, label=r'$0.1\%$ error level', color='green')
ax.set_xlabel('%s'%vary_str)
ax.set_title(suite+' '+ylabel)
ax.legend()
#ax.set_xticklabels(xlabel)
plt.savefig('/home/cemoser/Repositories/ostrich/Tests/Figures/Param_space/errs_3d_'+suite+'_'+vary_str+'_'+prof+'.png',bbox_inches='tight')
plt.close()

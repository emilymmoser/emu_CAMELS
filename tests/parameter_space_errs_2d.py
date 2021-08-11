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
suite='SIMBA'
vary_str=sys.argv[1]
mass_str=sys.argv[2] #warning: 12-12.2 isn't cut at x=0.01, so has more radial points
prof=sys.argv[3]

mass=fs.mass
#mass_str=fs.mass_str
snap=fs.snap
z=fs.choose_redshift(suite)
vary,sims=fs.choose_vary(vary_str)
samples=fs.cartesian_prod(vary,z)
nsamp=samples.shape[0]
usecols,ylabel=fs.choose_profile(prof)

x,y,errup,errlow,stddev=fs.load_profiles_2D(usecols,home,suite,sims,snap,mass_str)
y=np.transpose(y)
errup=np.transpose(errup)
errlow=np.transpose(errlow)
stddev=np.transpose(stddev)

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


print("errs shape",np.shape(errs_drop_1))

def chunks(indices,n):
    return [indices[i:i+n] for i in range(0,len(indices),n)]

total_indices=np.linspace(0,nsamp-1,nsamp,dtype='int')
print("total indices",len(total_indices))
chunked=chunks(total_indices,len(snap))
print("split into chunks of size len(snap)",np.shape(chunked))


#for 2D, each of these should be 10 redshifts*13 radial points=130
#for 3d, each of these 10 redshifts*13 radial*7 masses=910?
mean_radial_errs=[]
for i in range(len(vary)):
    idx=chunked[i]
    new_errs=errs_drop_1[:,idx]
    new_errs=np.reshape(new_errs,-1)
    mean_radial_errs.append(np.mean(np.log10(new_errs[new_errs>0])))

fig=plt.figure()
ax=plt.gca()
ax.scatter(vary,mean_radial_errs)
ax.set_ylabel('log (percent error)')
ax.axhline(-1, label=r'$10\%$ error level', color='red')
ax.axhline(-2, label=r'$1\%$ error level', color='orange')
ax.axhline(-3, label=r'$0.1\%$ error level', color='green')
ax.set_xlabel('%s'%vary_str)
ax.legend()
plt.savefig('/home/cemoser/Repositories/ostrich/Tests/Figures/errs_paramspace_2d_'+suite+'_'+vary_str+'.png',bbox_inches='tight')
plt.close()

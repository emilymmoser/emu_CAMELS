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


X,Y=np.meshgrid(vary,z)
print("Or if we do the meshgrid way")
print("X (ASN1)",np.shape(X))
print("Y (z)",np.shape(Y))

errs_mean=np.mean(errs_drop_1,axis=0)
errs_reshaped=errs_mean.reshape(len(vary),len(snap))
errs=np.transpose(errs_reshaped)

print("max err",np.log10(errs.max()))

'''
print(np.shape(errs))

for q in range(10):
    vary_idx=random.randint(0,len(vary)-1)
    z_idx=random.randint(0,len(z)-1)
    print("a =",vary_idx,"corresponds to %s = %f"%(vary_str,vary[vary_idx]))
    print("b =",z_idx,"corresponds to z =",z[z_idx])
    index=fs.retrieve_index_2D(vary_idx,z_idx)
    print("In our samples array, this would have been index",index) 
    print("In the old version of the errs_drop_1 array, the value for this index was",np.mean(errs_drop_1[:,index]))
    print("In the new version, the value for index %i,%i is"%(vary_idx,z_idx),errs[z_idx,vary_idx])
    print("-------------")
'''
fig=plt.figure()
left,bottom,width,height=0.1,0.1,0.8,0.8
ax=fig.add_axes([left,bottom,width,height])
levels=[-6,-5,-4,-3,-2,-1,0]
c=('darkgreen','darkgreen','darkgreen','green','yellow','red','red')
cp=ax.contourf(X,Y,np.log10(np.abs(errs)),levels,colors=c)
#cp=ax.contourf(X,Y,np.log10(np.abs(errs)),levels,cmap='RdYlGn')

#norm=matplotlib.colors.Normalize(vmin=cp.values.min(),vmax=cp.values.max())
#sm=plt.cm.ScalarMappable(norm=norm,cmap=cp.cmap)
#sm.set_array([])
#fig.colorbar(sm,ticks=cp.levels,label='log (percent err)')
plt.colorbar(cp,label='log (percent err)')
ax.set_xlabel('%s'%vary_str)
ax.set_ylabel('z')
plt.savefig('/home/cemoser/Repositories/ostrich/Tests/Figures/contour_test_'+suite+'_'+vary_str+'.png',bbox_inches='tight') 




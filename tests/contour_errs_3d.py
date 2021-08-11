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

print("y shape",np.shape(y))
print("samples shape",np.shape(samples))
print("nsamp",nsamp)
print("errs shape",np.shape(errs_drop_1))
np.savetxt('errs_drop1_emulate3d_'+suite+'_'+vary_str+'_'+prof+'.txt',errs_drop_1)
np.savetxt('emulated_drop1_emulate3d_'+suite+'_'+vary_str+'_'+prof+'.txt',emulated_drop_1)
'''

X,Y=np.meshgrid(z,mass)
print("X",np.shape(X))
print("Y",np.shape(Y))

errs_drop_1=np.genfromtxt('./Error_histograms/errs_drop1_'+suite+'_'+vary_str+'_'+prof+'_4Mbins_RbfInterp_lin.txt')
print("original errs",np.shape(errs_drop_1))
#errs_mean_rad=np.mean(errs_drop_1,axis=0)
#print("errs mean",np.shape(errs_mean_rad))

errs_reshaped=errs_drop_1.reshape(len(x),len(vary),len(snap),len(mass))
print("errs reshaped",np.shape(errs_reshaped))
#errs_mean_rad=np.mean(errs_reshaped,axis=0)
#print("errs mean rad",np.shape(errs_mean_rad))
#errs_mean_vary=np.mean(errs_reshaped,axis=0)


errs=np.transpose(errs_reshaped)
#print("max err",np.log10(errs.max()))
print("final errs",np.shape(errs))
'''
for q in range(2):
    vary_idx=random.randint(0,len(vary)-1)
    z_idx=random.randint(0,len(z)-1)
    mass_idx=random.randint(0,len(mass)-1)
    print("a =",vary_idx,"corresponds to %s = %f"%(vary_str,vary[vary_idx]))
    print("b =",z_idx,"corresponds to z =",z[z_idx])
    print("c =",mass_idx,"corresponds to M =",mass[mass_idx])
    index=fs.retrieve_index_3D(vary_idx,z_idx,mass_idx)
    print("In our samples array, this would have been index",index) 
    print("In the old version of the errs_drop_1 array, the value for this index was",np.mean(errs_drop_1[:,index]))
    print("In the new version, the value for index %i,%i,%i is"%(vary_idx,z_idx,mass_idx),np.mean(errs[mass_idx,z_idx,vary_idx]))
    print("-------------")

'''
for f in range(len(vary)):
    fig=plt.figure()
    left,bottom,width,height=0.1,0.1,0.8,0.8
    ax=fig.add_axes([left,bottom,width,height])
    levels=[-6,-5,-4,-3,-2,-1,0]
    c=('darkgreen','darkgreen','darkgreen','green','yellow','red','red')
    errf=np.log10(np.abs(np.mean(errs[:,:,f],axis=2)))
    print(np.shape(errf))
    cp=ax.contourf(X,Y,errf,levels,colors=c)
    plt.colorbar(cp,label='log (percent err)')
    ax.set_xlabel('z')
    ax.set_ylabel('log(M)')
    ax.set_title(r'%s, %s = %.5f'%(suite,vary_str,vary[f]))
    plt.savefig('/home/cemoser/Repositories/ostrich/Tests/Figures/Param_space/contour_%s_%s_%s_%.5f.png'%(suite,prof,vary_str,vary[f]),bbox_inches='tight') 

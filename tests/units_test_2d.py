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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import functions as fs
import sys


home='/home/cemoser/Repositories/ostrich/Emulator_profiles/'
suite=sys.argv[1]
vary_str=sys.argv[2]
prof=sys.argv[3]
mass_str=sys.argv[4]

mass=fs.mass
#mass_str=fs.mass_str
snap=fs.snap
z=fs.choose_redshift(suite)
vary,sims=fs.choose_vary(vary_str)
samples=fs.cartesian_prod(vary,z) 
nsamp=samples.shape[0]

start=time.time()
usecols,ylabel=fs.choose_profile(prof)
x,y,errup,errlow,stddev=fs.load_profiles_2D(usecols,home,suite,sims,snap,mass_str)

y=np.transpose(y)
errup=np.transpose(errup)
errlow=np.transpose(errlow)
stddev=np.transpose(stddev)

funcs_str='linear'

emulator = ostrich.emulate.PcaEmulator.create_from_data(
    samples,
    y,
    ostrich.interpolate.RbfInterpolator,
    interpolator_kwargs={'function': funcs_str},
    num_components=12,
)


def get_errs(samps,data,true_coord,true_data):
    emulator = ostrich.emulate.PcaEmulator.create_from_data(
        samps,
        data.reshape(data.shape[0],-1),
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function': funcs_str},
        num_components=12,
    )
    emulated =emulator(true_coord)
    true_data=true_data
    emulated=emulated.reshape(len(data))
    print("true coord",true_coord)
    print("emulated",np.shape(emulated),emulated)
    print("true data",np.shape(true_data),true_data)
    err=((emulated - true_data)/true_data).squeeze()
    print("err",np.shape(err),err)
    print("log err",np.log10(np.abs(err)))
    return emulated,err


errs_drop_1 = np.zeros_like(y)
emulated_drop_1=np.zeros_like(y)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for i in range(nsamp):
        emulated_drop_1[:,i],errs_drop_1[:,i] = get_errs(
            np.delete(samples, i,0),
            np.delete(y,i,1),
            samples[i:i+1],
            y[:,i],
        )



'''
np.savetxt('/home/cemoser/Repositories/ostrich/Tests/units_test/errs_drop1_emulator2d_'+suite+'_'+prof+'_'+vary_str+'_'+mass_str+'.txt',errs_drop_1)
np.savetxt('/home/cemoser/Repositories/ostrich/Tests/units_test/emulated_drop1_emulator2d_'+suite+'_'+prof+'_'+vary_str+'_'+mass_str+'.txt',emulated_drop_1)
'''




'''
errs_drop_1=np.genfromtxt('/home/cemoser/Repositories/ostrich/Tests/Figures/emulator3D/units_test/errs_drop1_'+suite+'_'+prof+'_'+vary_str+'.txt')
emulated_drop_1=np.genfromtxt('/home/cemoser/Repositories/ostrich/Tests/Figures/emulator3D/units_test/emulated_drop1_'+suite+'_'+prof+'_'+vary_str+'.txt')



abs_errs=np.abs(errs_drop_1.flatten())
mean_err=np.mean(abs_errs)
std_err=np.std(abs_errs)

plt.hist(np.log10(abs_errs[abs_errs>0]), histtype='stepfilled', bins=100,alpha=0.6,label=r'$%.3f \pm %.3f$'%(mean_err,std_err))
plt.axvline(np.log10(mean_err),color='k')
plt.axvline(-1, label=r'$10\%$ error level', color='red')
plt.axvline(-2, label=r'$1\%$ error level', color='orange')
plt.axvline(-3, label=r'$0.1\%$ error level', color='green')
plt.xlabel('log (percent error)',size=12)
plt.legend(loc='best')
plt.savefig('/home/cemoser/Repositories/ostrich/Tests/units_test/simba_emulator2d_'+prof+'_'+vary_str+'_'+mass_str+'.png',bbox_inches='tight') 
plt.close()


#radial error plots
for i in np.arange(nsamp-100):
    a,b=fs.deconstruct_2D(i) #a is the ASN1 index, b is the z index, c is the mass_index 
    newy=y[:,i]
    newerrup=errup[:,i]
    newerrlow=errlow[:,i]

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.loglog(x,10**newy,label='data for %s=%.2f, z=%.2f'%(vary_str,vary[a],z[b]))
    plt.loglog(x,10**emulated_drop_1[:,i],label='emu',color='k')
    plt.fill_between(x,10**(newy)+10**(newerrup),10**newy-10**newerrlow,facecolor='cornflowerblue',alpha=0.4)                                                                                                                   
    plt.xlabel('R (Mpc)',size=12)
    plt.ylabel(ylabel,size=12)
    plt.legend(loc='best',fontsize=8)

    plt.subplot(1,2,2)
    errs = np.log10(np.abs(errs_drop_1[:,i]))
    plt.semilogx(x,errs,linestyle='dashed',color='k')
    plt.axhline(-1, label=r'$10\%$ error level', color='red')
    plt.axhline(-2, label=r'$1\%$ error level', color='orange')
    plt.axhline(-3, label=r'$0.1\%$ error level', color='green')

    plt.ylabel(r'log($\%$ error)',size=12)
    plt.xlabel('R (Mpc)',size=12)
    plt.legend(loc='best',fontsize=8)
    plt.savefig('/home/cemoser/Repositories/ostrich/Tests/units_test/errs_emulator2d_'+suite+'_'+prof+'_'+vary_str+'_'+str(vary[a])+'_z'+str(z[b])+'_M'+mass_str+'.png',bbox_inches='tight')
    plt.close()


end=time.time()
print("Total time: %.2f minutes, %.2f hours"%(((end-start)/60.),((end-start)/3600.)))
'''

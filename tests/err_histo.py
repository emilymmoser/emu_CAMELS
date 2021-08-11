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

mass=fs.mass
mass_str=fs.mass_str
snap=fs.snap
z=fs.choose_redshift(suite)
vary,sims=fs.choose_vary(vary_str)
samples=fs.cartesian_prod(vary,z,mass) 
print("samples",np.shape(samples))
nsamp=samples.shape[0]
print("nsamp",nsamp)

start=time.time()
usecols,ylabel=fs.choose_profile(prof)
x,y,errup,errlow,stddev=fs.load_profiles_3D(usecols,home,suite,sims,snap,mass_str,prof)

y=np.transpose(y)
errup=np.transpose(errup)
errlow=np.transpose(errlow)
stddev=np.transpose(stddev)

funcs=['multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate']
funcs_str=['multiquad','inv','gauss','lin','cubic','quint','thinp']

plt.figure(figsize=(6,4))
for count,val in enumerate(funcs):
    print("Now completing emulator for function",val)
    emulator = ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function': val},
        num_components=12,
    )


    def get_errs(samps,data,true_coord,true_data):
        emulator = ostrich.emulate.PcaEmulator.create_from_data(
            samps,
            data.reshape(data.shape[0],-1),
            ostrich.interpolate.RbfInterpolator,
            interpolator_kwargs={'function': val},
            num_components=12,
        )
        emulated = emulator(true_coord)
        emulated=emulated.reshape(len(data))
        emulated=10**emulated
        true_data=10**true_data
        err=((emulated - true_data)/true_data).squeeze()
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


    abs_errs=np.abs(errs_drop_1.flatten())
    log_errs=np.log10(abs_errs)
    mean_err=np.mean(log_errs)
    std_err=np.std(log_errs)

    np.savetxt('./Error_histograms/errs_drop1_'+suite+'_'+vary_str+'_'+prof+'_4Mbins_RbfInterp_'+funcs_str[count]+'.txt',errs_drop_1)
    np.savetxt('./Error_histograms/emulated_drop1_'+suite+'_'+vary_str+'_'+prof+'_4Mbins_RbfInterp_'+funcs_str[count]+'.txt',emulated_drop_1)

    f=open('./Error_histograms/errhist_'+suite+'_'+prof+'_'+vary_str+'_4Mbins_RbfInterp.txt','a')
    f.write('function %s has mean log(err) %.3f +- %.3f \n'%(funcs_str[count],mean_err,std_err))
    f.close()

    plt.hist(np.log10(abs_errs), histtype='stepfilled', bins=100,alpha=0.6,label=r'%s: $%.3f \pm %.3f$'%(funcs_str[count],mean_err,std_err))
    plt.axvline(mean_err,linestyle='dashed')
plt.axvline(-1, label=r'$10\%$ error level', color='red')
plt.axvline(-2, label=r'$1\%$ error level', color='orange')
plt.axvline(-3, label=r'$0.1\%$ error level', color='green')
plt.xlabel('log (percent error)',size=12)
plt.legend(loc='best')
plt.savefig('/home/cemoser/Repositories/ostrich/Tests/Error_histograms/'+suite+'_'+prof+'_'+vary_str+'_4Mbins.png',bbox_inches='tight') 



end=time.time()
print("Total time: %.2f minutes, %.2f hours"%(((end-start)/60.),((end-start)/3600.)))

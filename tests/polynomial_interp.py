import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate
import numpy as np
import warnings
import random
import time
import ostrich.emulate
import ostrich.interpolate
import sklearn.gaussian_process.kernels as skgp_kernels
import helper_functions as fs
import sys

home='/home/cemoser/Repositories/emu_CAMELS/emulator_profiles/'
suite=sys.argv[1]
vary_str=sys.argv[2]
prof=sys.argv[3]
interp=sys.argv[4]


mass=fs.mass
mass_str=fs.mass_str
snap=fs.snap
z=fs.choose_redshift(suite)
vary,sims=fs.choose_vary(vary_str)
samples=fs.cartesian_prod(vary,z,mass)
nsamp=samples.shape[0]

start=time.time()
usecols,ylabel=fs.choose_profile(prof)
x,y,errup,errlow,stddev=fs.load_profiles_3D(usecols,home,suite,sims,snap,mass_str,prof)

y=np.transpose(y)
errup=np.transpose(errup)
errlow=np.transpose(errlow)
stddev=np.transpose(stddev)

if interp=='GP':

    def get_errs(samps,data,true_coord,true_data,kernel):
        emulator = ostrich.emulate.PcaEmulator.create_from_data(
            samps,
            data.reshape(data.shape[0],-1),
            ostrich.interpolate.GaussianProcessInterpolator,
            interpolator_kwargs={'kernel': kernel},
            num_components=12,
        )
        emulated = emulator(true_coord)
        emulated=emulated.reshape(len(data))
        emulated=10**emulated
        true_data=10**true_data
        err=((emulated - true_data)/true_data).squeeze()
        return emulated,err
    funcs=[skgp_kernels.Matern(),skgp_kernels.RBF(),skgp_kernels.ConstantKernel(),skgp_kernels.RationalQuadratic(),skgp_kernels.DotProduct()] #expsinsq doesn't work, constant is horrible
    funcs_str=['GP Matern','GP RBF','GP Constant','GP RatQuad','GP DotProd']
    interpolator=ostrich.interpolate.GaussianProcessInterpolator
    interpkw='kernel'

elif interp=='rbf':
    def get_errs(samps,data,true_coord,true_data,function):
        emulator = ostrich.emulate.PcaEmulator.create_from_data(
            samps,
            data.reshape(data.shape[0],-1),
            ostrich.interpolate.RbfInterpolator,
            interpolator_kwargs={'function': function},
            num_components=12,
        )
        emulated = emulator(true_coord)
        emulated=emulated.reshape(len(data))
        emulated=10**emulated
        true_data=10**true_data
        err=((emulated - true_data)/true_data).squeeze()
        return emulated,err
    funcs=['multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate'] #gauss consistently the worst
    funcs_str=['multiquad','inv','gauss','lin','cubic','quint','thinp']
    interpolator=ostrich.interpolate.RbfInterpolator
    interpkw='function'

elif interp=='poly':
    def get_errs(samps,data,true_coord,true_data,degree):
        emulator = ostrich.emulate.PcaEmulator.create_from_data(
            samps,
            data.reshape(data.shape[0],-1),
            ostrich.interpolate.PolynomialInterpolator,
            interpolator_kwargs={'degree': degree},
            num_components=12,
        )
        emulated = emulator(true_coord)
        emulated=emulated.reshape(len(data))
        emulated=10**emulated
        true_data=10**true_data
        err=((emulated - true_data)/true_data).squeeze()
        return emulated,err

    funcs=[3,4,5,6,7,8]
    funcs_str=['3','4','5','6','7','8']
    interpolator=ostrich.interpolate.PolynomialInterpolator
    interpkw='degree'

cmap=cm.get_cmap('viridis',len(funcs))
colors=cmap.colors

plt.figure(figsize=(6,4))
for count,val in enumerate(funcs):
    
    #GaussianProcessInterpolator, Rbf, Polynomial
    emulator = ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        interpolator,
        interpolator_kwargs={interpkw: val},
        num_components=12,
    )

    errs_drop_1 = np.zeros_like(y)
    emulated_drop_1=np.zeros_like(y)
    start=time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(nsamp):
            emulated_drop_1[:,i],errs_drop_1[:,i] = get_errs(
                np.delete(samples, i,0),
                np.delete(y,i,1),
                samples[i:i+1],
                y[:,i],val,
            )

    np.savetxt('./Interp_test/errs_drop1_'+suite+'_'+vary_str+'_'+prof+'_'+interp+'_'+funcs_str[count]+'.txt',errs_drop_1)
    np.savetxt('./Interp_test/emulated_drop1_'+suite+'_'+vary_str+'_'+prof+'_'+interp+'_'+funcs_str[count]+'.txt',emulated_drop_1)

    abs_errs=np.abs(errs_drop_1.flatten())
    log_errs=np.log10(abs_errs)
    mean_err=np.mean(log_errs)
    std_err=np.std(log_errs)
    plt.hist(log_errs, color=colors[count],histtype='stepfilled', alpha=0.6,bins=100,label=r'%s : $%.2f \pm %.2f$'%(funcs_str[count],mean_err,std_err))
    plt.axvline(mean_err,color='k')
plt.axvline(-1, label=r'$10\%$ error level', color='red')
plt.axvline(-2, label=r'$1\%$ error level', color='orange')
plt.axvline(-3, label=r'$0.1\%$ error level', color='green')
plt.legend(loc='best')
plt.savefig('./Interp_test/'+suite+'_'+interp+'_'+interpkw+'_'+vary_str+'_'+prof+'.png',bbox_inches='tight')

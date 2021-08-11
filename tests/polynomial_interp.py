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

time1=time.time()

home='/home/cemoser/Repositories/ostrich/Emulator_profiles/'
suite='SIMBA'
vary_str=sys.argv[1]
mass_str=sys.argv[2]

ASN1=np.array([0.25000,0.32988,0.43528,0.57435,0.75786,1.00000,1.31951,1.74110,2.29740,3.03143,4.00000])
ASN2=np.array([0.50000,0.57435,0.65975,0.75786,0.87055,1.00000,1.14870,1.31951,1.51572,1.74110,2.00000])
z_sim=np.array([0.00000,0.04896,0.10033,0.15420,0.21072,0.27000,0.33218,0.39741,0.46584,0.53761])
z_tng=np.array([0.00000,0.04852,0.10005,0.15412,0.21012,0.26959,0.33198,0.39661,0.46525,0.53726])
snap=['033','032','031','030','029','028','027','026','025','024']


if suite=='IllustrisTNG':
    z=z_tng
else:
    z=z_sim 
    
if vary_str=='ASN1':
    vary=ASN1
    nums=np.linspace(22,32,11,dtype='int')
elif vary_str=='AAGN1':
    vary=ASN1
    nums=np.linspace(33,43,11,dtype='int')
elif vary_str=='ASN2':
    vary=ASN2
    nums=np.linspace(44,54,11,dtype='int')
elif vary_str=='AAGN2':
    vary=ASN2
    nums=np.linspace(55,65,11,dtype='int')
    
sims=['1P_'+str(n) for n in nums]

samples=fs.cartesian_prod(vary,z) 
n_2d=samples.shape[0]

y=[]
yerr=[]
for s in np.arange(len(sims)):
    for n in np.arange(len(snap)):
        x,ysn,yerrsn=np.loadtxt(home+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_uw_'+mass_str+'.txt',usecols=(0,1,4),unpack=True)
        xcut,ysn=fs.inner_cut_1D(0.01,x,ysn)
        xcut,yerrsn=fs.inner_cut_1D(0.01,x,yerrsn)
        y.append(np.log10(ysn))
        yerr.append(yerrsn)

y=np.array(y)
y=np.transpose(y)
yerr=np.array(yerr)
yerr=np.transpose(yerr)
x=xcut


def get_errs_skgp(samps,data,true_coord,true_data,kernel):
    emulator = ostrich.emulate.PcaEmulator.create_from_data(
        samps,
        data.reshape(data.shape[0],-1),
        ostrich.interpolate.GaussianProcessInterpolator,
        interpolator_kwargs={'kernel': kernel},
        num_components=12,
    )
    emulated = emulator(true_coord)
    emulated=emulated.reshape(len(data))
    err=((emulated - true_data)/true_data).squeeze()
    return emulated,err

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

def get_errs_poly(samps,data,true_coord,true_data,degree):
    emulator = ostrich.emulate.PcaEmulator.create_from_data(
        samps,
        data.reshape(data.shape[0],-1),
        ostrich.interpolate.PolynomialInterpolator,
        interpolator_kwargs={'degree': degree},
        num_components=12,
    )
    emulated = emulator(true_coord)
    emulated=emulated.reshape(len(data))
    err=((emulated - true_data)/true_data).squeeze()
    return emulated,err


#kernels=[skgp_kernels.Matern(),skgp_kernels.RBF(),skgp_kernels.ConstantKernel(),skgp_kernels.RationalQuadratic(),skgp_kernels.DotProduct()]
#kernels_str=['GP Matern','GP RBF','GP Constant','GP RatQuad','GP DotProd']
#funcs=['multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate']
#funcs_str=['multiquad','inv','gauss','lin','cubic','quint','thinp']
#funcs=['cubic']
#funcs_str=['cubic']
kernels=[skgp_kernels.RationalQuadratic()]
kernels_str=['GP RatQuad']

#degrees=[3,4,5,6,7,8,9,10]
#degrees_str=['3','4','5','6','7','8','9','10']
#degrees=[8]
#degrees_str=['8']


cmap=cm.get_cmap('viridis',len(kernels))
colors=cmap.colors

plt.figure(figsize=(6,4))
for count,val in enumerate(kernels):

    #GaussianProcessInterpolator, Rbf, Polynomial
    emulator = ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.GaussianProcessInterpolator,
        interpolator_kwargs={'kernel': val},
        num_components=12,
    )

    errs_drop_1 = np.zeros_like(y)
    emulated_drop_1=np.zeros_like(y)
    start=time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(n_2d):
            emulated_drop_1[:,i],errs_drop_1[:,i] = get_errs_skgp(
                np.delete(samples, i,0),
                np.delete(y,i,1),
                samples[i:i+1],
                y[:,i],val,
            )

    abs_errs=np.abs(errs_drop_1.flatten())
    mean_err=np.mean(abs_errs)
    std_err=np.std(abs_errs)
    plt.hist(np.log10(abs_errs[abs_errs>0]), color=colors[count], histtype='stepfilled', alpha=0.6,bins=100,label=r'%s : $%.2f \pm %.2f$'%(kernels_str[count],mean_err,std_err))
    plt.axvline(np.log10(mean_err),color='k')
plt.axvline(-1, label=r'$10\%$ error level', color='red')
plt.axvline(-2, label=r'$1\%$ error level', color='orange')
plt.axvline(-3, label=r'$0.1\%$ error level', color='green')
plt.legend(loc='best')
#plt.savefig('/home/cemoser/Repositories/ostrich/Tests/Figures/simba_poly_degrees_'+mass_str+'_'+vary_str+'.png',bbox_inches='tight')

print(n_2d)
for i in np.arange(n_2d):

    a,b=fs.deconstruct_2D(i) #a is the ASN1 index, b is the z index
    newx=x 
    newy=y[:,i]
    newyerr=yerr[:,i]
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    
    plt.loglog(newx,10**newy,label='data for %s=%.2f, z=%.2f'%(vary_str,vary[a],z[b]))
    plt.loglog(newx,10**emulated_drop_1[:,i],label='emu',color='k')
    plt.fill_between(newx,10**(newy+newyerr),10**(newy-newyerr),facecolor='cornflowerblue',alpha=0.4)
    #if we had errup and errlow values, plot like this
    #plt.fill_between(newx,10**(newy)+10**(errup),10**newy-10**errlow)
    plt.xlabel('R (Mpc)',size=12)
    plt.ylabel(r'$\rho$',size=12)
    plt.legend(loc='best',fontsize=8)
    
    plt.subplot(1,2,2)
    errs = np.log10(np.abs(errs_drop_1[:,i]))
    plt.semilogx(newx,errs,linestyle='dashed',color='k')
    plt.axhline(-1, label=r'$10\%$ error level', color='red')
    plt.axhline(-2, label=r'$1\%$ error level', color='orange')
    plt.axhline(-3, label=r'$0.1\%$ error level', color='green')
    
    plt.ylabel('log(percent error)',size=12)
    plt.xlabel('R (Mpc)',size=12)
    plt.legend(loc='best',fontsize=8)
    plt.savefig('/home/cemoser/Repositories/ostrich/Tests/Figures/GP_interp/errs_GP_RatQuad_'+suite+'_'+vary_str+'_'+mass_str+'_'+str(vary[a])+'_z'+str(z[b])+'.png',bbox_inches='tight')
    plt.close()

time2=time.time()

print("Total time for script: %.2f minutes"%((time2-time1)/60.))

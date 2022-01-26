import numpy as np
import ostrich.emulate
import ostrich.interpolate
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import warnings


z_sim=np.array([0.00000,0.04896,0.10033,0.15420,0.21072,0.27000,0.33218,0.39741,0.46584,0.53761])
z_tng=np.array([0.00000,0.04852,0.10005,0.15412,0.21012,0.26959,0.33198,0.39661,0.46525,0.53726])

#NOTE: I only have the profiles for a single mass bin and redshift. It is straightforward to add more to the lines below in variables snap and mass, but the samples_6d method would also need to be changed to reflect the larger number of samples.
snap=['024']
mass=np.array([11.25])
mass_str=np.array(['11-11.5'])
mass_range_latex=np.array(['$11 \leq \log_{10}(M_{200c}/M_\odot) \leq 11.5$'])

usecols_dict={'rho_mean':(0,1),'rho_med':(0,5),'pth_mean':(0,6),'pth_med':(0,10),'metal_mean':(0,11),'metal_med':(0,15),'temp_mean':(0,16),'temp_med':(0,20)}
ylabel_3d_dict={'rho_mean':r'$\rho_{mean} (g/cm^3)$','rho_med':r'$\rho_{med} (g/cm^3)$','pth_mean':r'$P_{th,mean} (g/cm/s^2)$','pth_med':r'$P_{th,med} (g/cm/s^2)$','metal_mean':r'$\frac{Z_{mean}}{Z_{tot}}$','metal_med':r'$\frac{Z_{med}}{Z_{tot}}$','temp_mean':r'$T_{gas,mean} (K)$','temp_med':r'$T_{gas,med} (K)$'}
A_param_latex_dict={'ASN1':r'$A_{SN1}$','ASN2':r'$A_{SN2}$', 'AAGN1':r'$A_{AGN1}$','AAGN2':r'$A_{AGN2}$'}

def cgs_units(prof,arr):
    if prof=='rho_mean' or prof=='rho_med':
        #input rho in Msol/kpc^3
        rho=arr*u.solMass/u.kpc/u.kpc/u.kpc
        arr=rho.cgs
    elif prof=='pth_mean' or prof=='pth_med':
        #input pth in Msol/kpc/s^2
        pth=arr*u.solMass/u.kpc/(u.s*u.s)
        arr=pth.to(u.dyne/(u.cm*u.cm))
    return arr.value


def choose_redshift(suite):
    if suite=='IllustrisTNG':
        z=z_tng
    elif suite=='SIMBA':
        z=z_sim
    return z

def inner_cut(inner_cut,x,arr):
    idx=np.where(x >= inner_cut)
    idx=np.array(idx[0])
    x,arr=x[idx],arr[:,idx]
    return x,arr

def inner_cut_1D(inner_cut,x,arr):
    idx=np.where(x >= inner_cut)
    idx=np.array(idx[0])
    x,arr=x[idx],arr[idx]
    return x,arr

def cartesian_prod(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

def get_errs_3D(data,emulated):
    diff=(data-emulated)/data
    return (np.log10(np.abs(diff)))
    
#----------------------------------------------
#general emulator functions
def load_profiles_3D(usecols,home,suite,sims,snap,mass_str,prof):
    y=[]
    for s in np.arange(len(sims)):
        for n in np.arange(len(snap)):
            for m in np.arange(len(mass)):
                f=home+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_uw_'+mass_str[m]+'.txt'
                x,yi=np.loadtxt(f,usecols=usecols,unpack=True)

                if prof[:3]=='rho' or prof[:3]=='pth':
                    yi=cgs_units(prof,yi)
                y.append(np.log10(yi))

    y=np.array(y)
    return x,y

def samples_6d(omegam,sigma8,ASN1,AAGN1,ASN2,AAGN2):
    samples=[]
    for count,val in enumerate(omegam):
        samples.append([val,sigma8[count],ASN1[count],AAGN1[count],ASN2[count],AAGN2[count]])
    return np.array(samples)

def build_emulator_3D(home,suite,prof,func_str,omegam,sigma8,ASN1,AAGN1,ASN2,AAGN2):
    z=choose_redshift(suite)
    z=z[-1] #hard-coded for single redshift
    nums=np.linspace(0,999,1000,dtype='int')
    sims=['LH_'+str(i) for i in nums]

    samples=samples_6d(omegam,sigma8,ASN1,AAGN1,ASN2,AAGN2) #this would change for more masses/reds
    nsamp=samples.shape[0]

    usecols=usecols_dict[prof]
    x,y=load_profiles_3D(usecols,home,suite,sims,snap,mass_str,prof)
    y=np.transpose(y)

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=12)
    return samples,x,y,emulator


#plotting and testing functions
def get_errs_drop1(samps,data,true_coord,true_data):
    emulator = ostrich.emulate.PcaEmulator.create_from_data(
        samps,
        data.reshape(data.shape[0],-1),
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function': 'linear'},
        num_components=12,
    )
    emulated = emulator(true_coord)
    emulated=emulated.reshape(len(data))
    emulated=10**emulated
    true_data=10**true_data
    err=((emulated - true_data)/true_data).squeeze()
    return emulated,err

def drop1_test(y,nsamp,samples):
    errs_drop1=np.zeros_like(y)
    emulated_drop1=np.zeros_like(y)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(nsamp):
            emulated_drop1[:,i],errs_drop1[:,i]=get_errs_drop1(
                np.delete(samples,i,0),
                np.delete(y,i,1),
                samples[i:i+1],
                y[:,i],
            )

    return errs_drop1,emulated_drop1

def plot_drop1_test(x,y,emulated,errs,ylabel,legend_label):
    fig=plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.loglog(x,10**y,label=legend_label)
    plt.loglog(x,emulated,label='emu',color='k')
    plt.xlabel('R (Mpc)',size=12)
    plt.ylabel(ylabel,size=12)
    plt.legend(loc='best',fontsize=8)

    plt.subplot(1,2,2)
    plt.semilogx(x,errs,linestyle='dashed',color='k')
    plt.axhline(-1, label=r'$10\%$ error level', color='red')
    plt.axhline(-2, label=r'$1\%$ error level', color='orange')
    plt.axhline(-3, label=r'$0.1\%$ error level', color='green')
    plt.ylabel(r'log($\%$ error)',size=12)
    plt.xlabel('R (Mpc)',size=12)
    plt.legend(loc='best',fontsize=8)
    return fig

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique=[(h,l) for i, (h,l) in enumerate(zip(handles,labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def plot_drop1_percent_err_LH(x,y,emulated,errs,ylabel,title):
    fig=plt.figure(figsize=(6,8))
    gs=gridspec.GridSpec(2,1,height_ratios=[2,1])
    ax0=plt.subplot(gs[0])
    ax1=plt.subplot(gs[1])
    plt.setp(ax0.get_xticklabels(),visible=False)

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('R (Mpc)',size=12)

    y=10**y
    errs=100.*np.abs(errs)

    ax0.plot(x,emulated,linestyle='dashed',label='emulated')
    ax0.plot(x,y,label='data',linewidth=1)
    ax1.plot(x,errs,linewidth=1)
    ax1.set_ylabel(r'$\%$ error')
    ax0.set_ylabel(ylabel,size=12)
    ax0.tick_params(which='both',direction='in')
    ax1.tick_params(which='both',direction='in')
    plt.setp(ax0.get_xticklabels(),Fontsize=12)
    plt.setp(ax0.get_yticklabels(),Fontsize=12)
    plt.setp(ax1.get_xticklabels(),Fontsize=12)
    plt.setp(ax1.get_yticklabels(),Fontsize=12)

    legend_without_duplicate_labels(ax0)
    plt.suptitle(title)
    gs.tight_layout(fig,rect=[0,0,1,0.97])
    return fig

def choose_ylabel(prof,dimension):
    if dimension==2:
        ylabel=ylabel_2d_dict[prof]
    elif dimension==3:
        ylabel=ylabel_3d_dict[prof]
    return ylabel

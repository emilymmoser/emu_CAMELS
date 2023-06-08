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
#snap=['033','032','031','030','029','028','027','026','025','024']
snap = ['024']

mass=np.array([11.25,11.75])
mass_str=np.array(['11-11.5','11.5-12'])

usecols_dict={'rho_mean':(0,1),'rho_med':(0,5),'pth_mean':(0,6),'pth_med':(0,10),'metal_mean':(0,11),'metal_med':(0,15),'temp_mean':(0,16),'temp_med':(0,20)}
usecols_w_dict={'rho_mean':(0,1),'pth_mean':(0,5)}
ylabel_3d_dict={'rho_mean':r'$\rho_{mean} (g/cm^3)$','rho_med':r'$\rho_{med} (g/cm^3)$','pth_mean':r'$P_{th,mean} (g/cm/s^2)$','pth_med':r'$P_{th,med} (g/cm/s^2)$','metal_mean':r'$\frac{Z_{mean}}{Z_{tot}}$','metal_med':r'$\frac{Z_{med}}{Z_{tot}}$','temp_mean':r'$T_{gas,mean} (K)$','temp_med':r'$T_{gas,med} (K)$'}
ylabel_2d_dict={'rho_mean':r'$T_{kSZ} (\mu K)$','pth_mean':r'$T_{tSZ} (\mu K)$'}
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

def set_suite(suite):
    txt_file = 'CosmoAstroSeed_'+suite+'_reduced.txt'

    data = np.loadtxt(txt_file, dtype={'names': ('sim_name', 'omegam', 'sigma8', 'asn1', 'aagn1', 'asn2', 'aagn2', 'seed'),
                                   'formats': ('S10', float, float, float, float, float, float, int )} )

    Sim_name = data['sim_name']
    OmegaM = data['omegam']
    sigma8 = data['sigma8']
    ASN1 = data['asn1']
    ASN2 = data['asn2']
    AAGN1 = data['aagn1']
    AAGN2 = data['aagn2']

    return Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2

def LH_cartesian_prod(params, redshift, mass):
    arr_shape = (params.shape[0]*redshift.shape[0]* mass.shape[0],params.shape[1]+2)
    arr = np.zeros(arr_shape)
    for i in range(params.shape[0]):
        for j, z in enumerate(redshift):
            for k, m in enumerate(mass):
                index = k+mass.shape[0]*(j+redshift.shape[0]*i)
                arr[index,-1] = m
                arr[index,-2] = z
                for l, p in enumerate(params[i]):
                    arr[index,l] = p
    return arr

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

def build_emulator_3D(home,suite,prof,func_str):
    z=choose_redshift(suite)
    #these lines need to be adjusted if you have more than one redshift
    #but this will make the github version work for the currently uploaded profiles
    z=np.array([z[-1]])
    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)
    nums_tot=np.linspace(0,999,1000,dtype='int')

    #remove some sims for radial limits
    if suite == 'SIMBA':
        nums_remove = np.array([5,6,16,37,77,90,95,103,127,133,149,156,161,166,168,179,184,222,234,241,257,267,270,295,303,311,313,322,324,325,356,385,389,418,430,460,492,500,522,531,605,615,616,622,625,629,647,659,678,680,692,723,735,742,766,776,779,783,802,811,830,834,880,897,930,936,943,958,970,977])
    if suite == 'IllustrisTNG':
        nums_remove = np.array([43, 51, 75, 102, 111, 122, 123, 138, 145, 183, 207, 225, 233, 263, 298, 344, 372, 397, 439, 449, 453, 477, 484, 492, 505, 512, 539, 577, 584, 607, 611, 617, 646, 661, 675,713, 719, 726, 728, 743, 800, 801, 837, 841, 888, 898, 914, 921, 942, 948, 964])

    nums = [n for n in nums_tot if n not in nums_remove]
    sims=['LH_'+str(i) for i in nums]
    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod(params,z,mass)
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

#----------------------------------------
#CMASS emulator functions
mass_CMASS = np.array([12.25])	#just need something to put in for LH_cartesian_prod
mass_CMASS_str = np.array(['11.3-13.2'])
mass_CMASS_latex = np.array('$11.3 \leq \log_{10}(M_{200c}/M_\odot) \leq 13.2$')
def load_profiles_CMASS(usecols,home,suite,sims,prof):
    y=[]
    for s in np.arange(len(sims)):
        f=home+suite+'/'+suite+'_'+sims[s]+'_024_w.txt'
        x,yi=np.loadtxt(f,usecols=usecols,unpack=True)
        yi=cgs_units(prof,yi)
        y.append(np.log10(yi))

    y=np.array(y)
    return x,y

def build_emulator_CMASS(home,suite,prof,func_str):
    z=choose_redshift(suite)
    z=np.array([z[-1]])

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)
    nums_tot=np.linspace(0,999,1000,dtype='int')

    #remove some sims for radial limits   
    if suite == 'SIMBA':
        nums_remove = np.array([5,6,16,37,77,90,95,103,127,133,149,156,161,166,168,179,184,222,234,241,257,267,270,295,303,311,313,322,324,325,356,385,389,418,430,460,492,500,522,531,605,615,616,622,625,629,647,659,678,680,692,723,735,742,766,776,779,783,802,811,830,834,880,897,930,936,943,958,970,977])
    if suite == 'IllustrisTNG':
        nums_remove = np.array([43, 51, 75, 102, 111, 122, 123, 138, 145, 183, 207, 225, 233, 263, 298, 344, 372, 397, 439, 449, 453, 477, 484, 492, 505, 512, 539, 577, 584, 607, 611, 617, 646, 661, 675,713, 719, 726, 728, 743, 800, 801, 837, 841, 888, 898, 914, 921, 942, 948, 964])

    nums = [n for n in nums_tot if n not in nums_remove]
    sims=['LH_'+str(i) for i in nums]
    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod(params,z,mass_CMASS)
    nsamp=samples.shape[0]
    usecols=usecols_w_dict[prof]

    x,y=load_profiles_CMASS(usecols,home,suite,sims,prof)
    y=np.transpose(y)
    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=12)
    return samples,x,y,emulator

def compute_pm_profiles_CMASS(vary_str,A_emu,delta_thet,emulator,x,omega_m,sigma_8,z,M):
    if vary_str == 'ASN1':
        params_plus=[[omega_m,sigma_8,A_emu+delta_thet,1.0,1.0,1.0,z,M]]
        params_minus=[[omega_m,sigma_8,A_emu-delta_thet,1.0,1.0,1.0,z,M]]
    elif vary_str == 'AAGN1':
        params_plus=[[omega_m,sigma_8,1.0,A_emu+delta_thet,1.0,1.0,z,M]]
        params_minus=[[omega_m,sigma_8,1.0,A_emu-delta_thet,1.0,1.0,z,M]]
    elif vary_str == 'ASN2':
        params_plus=[[omega_m,sigma_8,1.0,1.0,A_emu+delta_thet,1.0,z,M]]
        params_minus=[[omega_m,sigma_8,1.0,1.0,A_emu-delta_thet,1.0,z,M]]
    elif vary_str == 'AAGN2':
        params_plus=[[omega_m,sigma_8,1.0,1.0,1.0,A_emu+delta_thet,z,M]]
        params_minus=[[omega_m,sigma_8,1.0,1.0,1.0,A_emu-delta_thet,z,M]]

    profile_plus=emulator(params_plus).reshape(len(x))
    profile_minus=emulator(params_minus).reshape(len(x))
    return profile_plus,profile_minus

#--------------------------------------------------

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

#derivative functions
def derivative(profile_up,profile_low,delta):
    deriv=(profile_up-profile_low)/(2.*delta)
    return deriv

def plot_derivatives(x,yf,yp,ym,yd,ylabel,title,dimension):
    fig=plt.figure(figsize=(6,8))
    gs=gridspec.GridSpec(2,1,height_ratios=[2,1])
    ax0=plt.subplot(gs[0])
    ax1=plt.subplot(gs[1])
    plt.setp(ax0.get_xticklabels(),visible=False)

    if dimension==3:
        ax0.set_xscale('log')
        ax1.set_xscale('log')
        ax1.set_xlabel('R (Mpc)',size=12)
    elif dimension==2:
        ax1.set_xlabel(r'$\theta$ (arcmin)')

    ax0.plot(x,yf,color='purple',label='fiducial')
    ax0.plot(x,yp,color='r',label='plus')
    ax0.plot(x,ym,color='b',label='minus')

    ax1.plot(x,yd,'-o')
    ax1.axhline(0,linestyle='dashed',color='gray',alpha=0.6
)
    ax1.set_ylabel('Derivative')
    ax0.set_ylabel(ylabel,size=12)
    ax0.legend()
    plt.suptitle(title)
    gs.tight_layout(fig,rect=[0,0,1,0.97])
    return fig

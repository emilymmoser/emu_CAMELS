import numpy as np
import ostrich.emulate
import ostrich.interpolate
from astropy import units as u
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import cm
import warnings

ASN1=np.array([0.25000,0.32988,0.43528,0.57435,0.75786,1.00000,1.31951,1.74110,2.29740,3.03143,4.00000])
ASN2=np.array([0.50000,0.57435,0.65975,0.75786,0.87055,1.00000,1.14870,1.31951,1.51572,1.74110,2.00000])
z_sim=np.array([0.00000,0.04896,0.10033,0.15420,0.21072,0.27000,0.33218,0.39741,0.46584,0.53761])
z_tng=np.array([0.00000,0.04852,0.10005,0.15412,0.21012,0.26959,0.33198,0.39661,0.46525,0.53726])
snap=['033','032','031','030','029','028','027','026','025','024']
AAGN1=ASN1
AAGN2=ASN2
mass=np.array([11.25,11.75,12.15,12.7])
mass_str=np.array(['11-11.5','11.5-12','12-12.3','12.3-13.1'])
mass_range_latex=np.array(['$11 \leq \log_{10}(M_{200c}/M_\odot) \leq 11.5$','$11.5 \leq \log_{10}(M_{200c}/M_\odot) \leq 12$','$12 \leq \log_{10}(M_{200c}/M_\odot) \leq 12.3$','$12.3 \leq \log_{10}(M_{200c}/M_\odot) \leq 13.1$'])

usecols_dict={'rho_mean':(0,1,2,3,4),'rho_med':(0,5,2,3,4),'pth_mean':(0,6,7,8,9),'pth_med':(0,10,7,8,9),'metal_mean':(0,11,12,13,14),'metal_med':(0,15,12,13,14),'temp_mean':(0,16,17,18,19),'temp_med':(0,20,17,18,19)}
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

def choose_vary(vary_str):
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
    return vary,sims

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

def retrieve_index_2D(a,b):
    index=a*len(snap)+b
    return index

def deconstruct_2D(index):
    a,b=divmod(index,len(snap))
    return a,b

def retrieve_index_3D(a,b,c):
    index=a*len(snap)*len(mass)+b*len(mass)+c
    return index

def deconstruct_3D(index):
    a,R=divmod(index,len(snap)*len(mass))
    b,c=divmod(R,len(mass))
    return a,b,c

def get_errs_3D(data,emulated):
    #emulated=emulated.reshape(len(data))
    diff=(data-emulated)/data
    return (np.log10(np.abs(diff)))
    
#----------------------------------------------
#general emulator functions
def load_profiles_3D(usecols,home,suite,sims,snap,mass_str,prof):
    y=[]
    errup=[]
    errlow=[]
    stddev=[]
    for s in np.arange(len(sims)):
        for n in np.arange(len(snap)):
            for m in np.arange(len(mass)):
                f=home+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_uw_'+mass_str[m]+'.txt'
                x,yi,errupi,errlowi,stddevi=np.loadtxt(f,usecols=usecols,unpack=True)

                if prof[:3]=='rho' or prof[:3]=='pth':
                    yi,errupi,errlowi=cgs_units(prof,yi),cgs_units(prof,errupi),cgs_units(prof,errlowi)
                y.append(np.log10(yi))
                errup.append(np.log10(errupi))
                errlow.append(np.log10(errlowi))
                stddev.append(stddevi)

    y,errup,errlow,stddev=np.array(y),np.array(errup),np.array(errlow),np.array(stddev)
    return x,y,errup,errlow,stddev

#NOTE: this is built-in with RbfInterpolator, make this an option at some point
def build_emulator_3D(home,suite,vary_str,prof,func_str):
    z=choose_redshift(suite)
    vary,sims=choose_vary(vary_str)

    samples=cartesian_prod(vary,z,mass)
    nsamp=samples.shape[0]

    usecols=usecols_dict[prof]
    x,y,errup,errlow,stddev=load_profiles_3D(usecols,home,suite,sims,snap,mass_str,prof)
    y=np.transpose(y)

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=12)
    return samples,x,y,emulator


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

def compute_weighted_profiles_pm(A_emu,delta_thet,z_emu,emulator,x): #take out high mass bin
    b_edges=np.array([12.11179316,12.46636941,12.91135125,13.42362312])#,13.98474899]) 
    b_cen=np.array([12.27689266, 12.67884686, 13.16053855])#, 13.69871423])
    p=np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01])#, 8.13760167e-02])
    w=[]
    for i in range(0,len(b_cen)):
        index=np.searchsorted(b_edges,b_cen[i])
        w.append(p[index-1])
    w=np.array(w,dtype='float')

    profiles_plus_uw=[]
    profiles_minus_uw=[]
    for m in b_cen:
        params_plus=[[A_emu+delta_thet,z_emu,m]]
        params_minus=[[A_emu-delta_thet,z_emu,m]]
        profiles_plus_uw.append(emulator(params_plus).reshape(len(x)))
        profiles_minus_uw.append(emulator(params_minus).reshape(len(x)))

    profile_plus_w=np.average(profiles_plus_uw,weights=w,axis=0)
    profile_minus_w=np.average(profiles_minus_uw,weights=w,axis=0)
    return profile_plus_w,profile_minus_w

def compute_unweighted_profiles_pm(A_emu,delta_thet,z_emu,emulator,x,M):
    params_plus=[[A_emu+delta_thet,z_emu,M]]
    params_minus=[[A_emu-delta_thet,z_emu,M]]

    profile_plus=emulator(params_plus).reshape(len(x))
    profile_minus=emulator(params_minus).reshape(len(x))
    return profile_plus,profile_minus

#CMASS emulator scripts
def load_profiles_CMASS(usecols,home,suite,sims,snap,prof):
    y=[]
    for s in np.arange(len(sims)):
        f=home+suite+'/'+suite+'_'+sims[s]+'_'+snap+'_w.txt'
        x,yi=np.loadtxt(f,usecols=usecols,unpack=True)
        yi=cgs_units(prof,yi)
        y.append(np.log10(yi))

    y=np.array(y)
    return x,y

def build_emulator_CMASS(home,suite,vary_str,prof,func_str):
    z=choose_redshift(suite)
    z=z[-1]
    vary,sims=choose_vary(vary_str)
    snap='024'
    samples=vary
    nsamp=len(samples)

    usecols=usecols_w_dict[prof]
    x,y=load_profiles_CMASS(usecols,home,suite,sims,snap,prof)
    y=np.transpose(y)

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=12)
    return samples,x,y,emulator

def compute_pm_profiles_CMASS(A_emu,delta_thet,emulator,x):
    params_plus=[[A_emu+delta_thet]]
    params_minus=[[A_emu-delta_thet]]

    profile_plus=emulator(params_plus).reshape(len(x))
    profile_minus=emulator(params_minus).reshape(len(x))
    return profile_plus,profile_minus


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

def plot_drop1_percent_err(x,y,emulated,errs,vary,vary_str,ylabel,title):
    fig=plt.figure(figsize=(6,8))
    gs=gridspec.GridSpec(2,1,height_ratios=[2,1])
    ax0=plt.subplot(gs[0])
    ax1=plt.subplot(gs[1])
    plt.setp(ax0.get_xticklabels(),visible=False)

    ax0.set_xscale('log')
    ax0.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('R (Mpc)',size=12)

    cmap=cm.get_cmap('viridis',len(vary))
    colors=cmap.colors
    for i in np.arange(len(vary)):
        yi=10**y[i,:]
        emulatedi=emulated[i,:]
        errsi=100.*np.abs(errs[i,:])

        ax0.plot(x,emulatedi,color=colors[i],linestyle='dashed',label='emulated')
        ax0.plot(x,yi,color=colors[i],label='%s = %.2f'%(vary_str,vary[i]),linewidth=1)
        ax1.plot(x,errsi,color=colors[i],linewidth=1)
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

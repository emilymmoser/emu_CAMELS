import sys; sys.path.append("ostrich")
import numpy as np
import ostrich.emulate
import ostrich.interpolate
from astropy import units as u

ASN1=np.array([0.25000,0.32988,0.43528,0.57435,0.75786,1.00000,1.31951,1.74110,2.29740,3.03143,4.00000])
ASN2=np.array([0.50000,0.57435,0.65975,0.75786,0.87055,1.00000,1.14870,1.31951,1.51572,1.74110,2.00000])
z_sim=np.array([0.00000,0.04896,0.10033,0.15420,0.21072,0.27000,0.33218,0.39741,0.46584,0.53761])
z_tng=np.array([0.00000,0.04852,0.10005,0.15412,0.21012,0.26959,0.33198,0.39661,0.46525,0.53726])
snap=['033','032','031','030','029','028','027','026','025','024']
AAGN1=ASN1
AAGN2=ASN2
mass=np.array([12.1,12.3,12.6,13.])
mass_str=np.array(['12-12.2','12.2-12.4','12.4-12.8','12.8-13.2'])

usecols_dict={'rho_mean':(0,1,2,3,4),'rho_med':(0,5,2,3,4),'pth_mean':(0,6,7,8,9),'pth_med':(0,10,7,8,9),'metal_mean':(0,11,12,13,14),'metal_med':(0,15,12,13,14),'temp_mean':(0,16,17,18,19),'temp_med':(0,20,17,18,19)}
ylabel_dict={'rho_mean':r'$\rho_{mean} (g/cm^3)$','rho_med':r'$\rho_{med} (g/cm^3)$','pth_mean':r'$P_{th,mean} (g/cm/s^2)$','pth_med':r'$P_{th,med} (g/cm/s^2)$','metal_mean':r'$\frac{Z_{mean}}{Z_{tot}}$','metal_med':r'$\frac{Z_{med}}{Z_{tot}}$','temp_mean':r'$T_{gas,mean} (K)$','temp_med':r'$T_{gas,med} (K)$'}

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

def choose_profile(prof):
    usecols=usecols_dict[prof]
    ylabel=ylabel_dict[prof]
    return usecols,ylabel

def load_profiles_2D(usecols,home,suite,sims,snap,mass_str,prof):
    y=[]
    errup=[]
    errlow=[]
    stddev=[]
    for s in np.arange(len(sims)):
        for n in np.arange(len(snap)):
            f=home+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_uw_'+mass_str+'.txt'
            x,yi,errupi,errlowi,stddevi=np.loadtxt(f,usecols=usecols,unpack=True)
            yi,errupi,errlowi=cgs_units(prof,yi),cgs_units(prof,errupi),cgs_units(prof,errlowi)
            y.append(np.log10(yi))
            errup.append(np.log10(errupi))
            errlow.append(np.log10(errlowi))
            stddev.append(stddevi)

    y,errup,errlow,stddev=np.array(y),np.array(errup),np.array(errlow),np.array(stddev)
    return x,y,errup,errlow,stddev

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

def get_errs_2D(data,emulated):
    diff=(data-emulated)/data
    return (np.log10(np.abs(diff)))

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
    

def build_emulator_2D(home,suite,vary_str,prof,mass_str,func_str): 
    z=choose_redshift(suite)
    vary,sims=choose_vary(vary_str)

    samples=cartesian_prod(vary,z) 
    nsamp=samples.shape[0]

    usecols,ylabel=choose_profile(prof)
    x,y,errup,errlow,stddev=load_profiles_2D(usecols,home,suite,sims,snap,mass_str,prof)
    y=np.transpose(y)
    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=12)
    return samples,x,y,emulator

#----------------------------------------------!!!!
#NOTE: this is built-in with RbfInterpolator, make this an option at some point
def build_emulator_3D(home,suite,vary_str,prof,func_str):
    z=choose_redshift(suite)
    vary,sims=choose_vary(vary_str)

    samples=cartesian_prod(vary,z,mass)
    nsamp=samples.shape[0]

    usecols,ylabel=choose_profile(prof)
    x,y,errup,errlow,stddev=load_profiles_3D(usecols,home,suite,sims,snap,mass_str,prof)
    y=np.transpose(y)

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=12)
    return samples,x,y,emulator

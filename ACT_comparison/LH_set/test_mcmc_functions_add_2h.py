import numpy as np
from astropy import units as u
import sys
sys.path.append('/home/cemoser/Projection_Codes/Mop-c-GT-copy/mopc')
import mopc_fft as mop
import ostrich.emulate
import ostrich.interpolate
from scipy import stats

z_sim=np.array([0.00000,0.04896,0.10033,0.15420,0.21072,0.27000,0.33218,0.39741,0.46584,0.53761])
z_tng=np.array([0.00000,0.04852,0.10005,0.15412,0.21012,0.26959,0.33198,0.39661,0.46525,0.53726])

#----------------projection, needs to be updated with everything from Moser23
home_mopc='/home/cemoser/Projection_Codes/Mop-c-GT-copy/'
ell2 = np.genfromtxt(home_mopc+'data/act_planck_s08_s18_cmb_f150_daynight_response_tsz.txt')[:,0]
res_150 = np.genfromtxt(home_mopc+'data/act_planck_s08_s18_cmb_f150_daynight_response_tsz.txt')[:,1]
nu=150.
z=0.54
sr2sqarcmin = 3282.8 * 60.**2


#updated beam
beam_new2_150 = np.load('Corrected_beam_150_v2.npy')
ell=beam_new2_150[0]
beam_150_ell=beam_new2_150[1]

def fBeamF_150(x):
    return np.interp(x,ell,beam_150_ell)
def respT_150(x):
    return np.interp(x,ell2,res_150)

def project_profiles(prof,theta,z,nu,beam,respT,x,profile,theta2h):
    if prof=='rho_mean':
        proj=mop.make_a_obs_profile_rho_array_ACT(theta,z,beam,x,profile,theta2h)
    elif prof=='pth_mean':
        proj=mop.make_a_obs_profile_pth_array_ACT(theta,z,nu,beam,respT,x,profile,theta2h)
    return proj


#----------emulator
usecols_w_dict={'rho_mean':(0,1),'pth_mean':(0,5)}
mass_CMASS = np.array([12.25])  #just need something to put in for LH_cartesian_prod
mass_CMASS_str = np.array(['11.3-13.2'])
#These will go into lnlike for our set params going into the emulator
omega_m_set = 0.3
sigma8_set = 0.8
z_set = 0.54
mass_set = 12.25
snap = '024'

def choose_redshift(suite):
    if suite=='IllustrisTNG':
        z=z_tng
    elif suite=='SIMBA':
        z=z_sim
    return z

def set_suite(suite):
    txt_file = 'CosmoAstroSeed_'+suite+'_reduced.txt' #reduced is cutting some of the sims

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

def build_emulator(home,suite,prof):
    z=choose_redshift(suite)
    z=np.array([z[-1]])

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)
    nums_tot=np.linspace(0,999,1000,dtype='int')

    #remove some sims for radial limits
    if suite == 'SIMBA':
        nums_remove = np.array([5,6,16,37,77,90,95,103,127,133,149,156,161,166,168,179,184,222,234,241,257,267,270,295,303,311,313,322,324,325,356,385,389,418,430,460,492,500,522,531,605,615,616,622,625,629,647,659,678,680,692,723,735,742,766,776,779,783,802,811,830,834,880,897,930,936,943,958,970,977])
    if suite == 'IllustrisTNG':
        nums_remove = np.array([43, 51, 75, 102, 111, 122, 123, 138, 145, 183, 207, 225, 233, 263, 298, 344, 372, 397, 439, 449, 453, 477, 484, 492, 505, 512, 539, 577, 584, 607, 611, 617, 646, 661, 675, 713, 719, 726, 728, 743, 800, 801, 837, 841, 888, 898, 914, 921, 942, 948, 964])
    
    nums = [n for n in nums_tot if n not in nums_remove]    
    sims=['LH_'+str(i) for i in nums]
    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod(params,z,mass_CMASS)
    nsamp=samples.shape[0]
    usecols=usecols_w_dict[prof]

    x,y=load_profiles(usecols,home,suite,sims,snap,prof)
    y=np.transpose(y)

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':'linear'},
        num_components=12)
    return samples,x,y,emulator

def load_profiles(usecols,home,suite,sims,snap,prof):
    #print("sims from load profiles",np.shape(sims),sims)
    y=[]
    for s in np.arange(len(sims)):
        f=home+suite+'/'+suite+'_'+sims[s]+'_024_w.txt'
        x,yi=np.loadtxt(f,usecols=usecols,unpack=True)
        yi=cgs_units(prof,yi)
        y.append(np.log10(yi))

    y=np.array(y)
    return x,y

def cgs_units(prof,arr):
    if prof=='rho_mean':
        rho=arr*u.solMass/u.kpc/u.kpc/u.kpc
        arr=rho.cgs
    elif prof=='pth_mean':
        pth=arr*u.solMass/u.kpc/(u.s*u.s)
        arr=pth.to(u.dyne/(u.cm*u.cm))
    return arr.value


#-------------- mcmc functions
#note 1.7 factor multiplied into rho_mean array
rescaling_factor_dict={'rho_mean':1.7*np.array([1.29434376,1.50186486,1.68729486,1.7872668,1.82233295,1.80844895,1.79453689,1.79890591,1.7996796]),
                       'pth_mean_remove_inner_bin':np.array([3.51593371,4.12355484,4.18725347,3.60379925,2.91902071,2.45097692,2.1838903,2.03076755]),
                       'pth_mean':np.array([2.92388319,3.51593371,4.12355484,4.18725347,3.60379925,2.91902071,2.45097692,2.1838903,2.03076755])}

def add_dust(tsz_profile): #tsz in muK*sr
    dust_profile=np.array([0.03611198,0.60664831,1.10251209,1.38862518,1.39781291,1.13080406,0.65623086,0.11062901,0.10156234])

    dust_profile/=sr2sqarcmin #convert back to muK*sr
    return tsz_profile+dust_profile

def lnprob(theta,thta_arc,proj,cov,x,prof,emulator):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    res=lp+lnlike(theta,thta_arc,proj,cov,x,prof,emulator)
    if np.isnan(res):
        return -np.inf
    return res


def lnlike(theta,thta_arc,proj,cov,x,prof,emulator):
    ASN1,AAGN1,ASN2,AAGN2,theta2h=theta
    y_proj=proj

    params=[[omega_m_set,sigma8_set,ASN1,AAGN1,ASN2,AAGN2,z_set,mass_set]]
    profile=emulator(params).reshape(len(x))
    profile=10**profile #undo the log

    model_proj=project_profiles(prof,thta_arc,z_set,nu,fBeamF_150,respT_150,x,profile,theta2h)

    rescaling_factor=rescaling_factor_dict[prof]
    model_proj*=rescaling_factor

    if prof=='pth_mean': 
        #add dust correction
        model_proj=add_dust(model_proj)

    diff=y_proj-model_proj
    var= -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    return var

def lnprior(theta):
    ASN1,AAGN1,ASN2,AAGN2,theta2h=theta
    #same priors as 1P
    if 0.25 < ASN1 < 4.0 and 0.25 < AAGN1 < 4.0 and 0.5 < ASN2 < 2.0 and 0.5 < AAGN2 < 2.0 and 0.01 < theta2h < 5.0:
        return 0.0
    return -np.inf

#------- mcmc helper functions
load_data_dict={'rho_mean':'ksz','pth_mean':'tsz'}
def load_ACT_data(home,prof):
    sr2sqarcmin = 3282.8 * 60.**2
    file_string=load_data_dict[prof]
    
    thta_act,act_data,act_err=np.loadtxt(home+'diskring_'+file_string+'_varweight_measured.txt',usecols=(0,1,2),unpack=True)

    cov=np.loadtxt(home+'cov_diskring_'+file_string+'_varweight_bootstrap.txt')
    
    cov_diag=np.sqrt(np.diag(cov))
    return thta_act,act_data,cov,cov_diag
    
def load_chains(home,suite,prof,identity,nwalkers,itr,ndim,cut):
    chain1=np.genfromtxt(home+'samples_total_'+suite+'_'+prof+'_run1_'+identity+'.txt')
    chain1_reshaped=chain1.reshape((nwalkers,itr,ndim))

    chain2=np.genfromtxt(home+'samples_total_'+suite+'_'+prof+'_run2_'+identity+'.txt')
    chain2_reshaped=chain2.reshape((nwalkers,itr,ndim))

    chain3=np.genfromtxt(home+'samples_total_'+suite+'_'+prof+'_run3_'+identity+'.txt')
    chain3_reshaped=chain3.reshape((nwalkers,itr,ndim))

    chains_all=np.concatenate((chain1_reshaped,chain2_reshaped,chain3_reshaped),axis=0)
    chains_all_cut=chains_all[:,cut:,:]
    chains_all_reshaped=chains_all.reshape((-1,ndim))
    chains_all_reshaped_cut=chains_all_cut.reshape((-1,ndim))
    return chains_all,chains_all_reshaped,chains_all_reshaped_cut

def load_logprob(home,suite,prof,identity,cut):
    logprob1=np.genfromtxt(home+'logprob_total_'+suite+'_'+prof+'_run1_'+identity+'.txt')
    logprob2=np.genfromtxt(home+'logprob_total_'+suite+'_'+prof+'_run2_'+identity+'.txt')
    logprob3=np.genfromtxt(home+'logprob_total_'+suite+'_'+prof+'_run3_'+identity+'.txt')

    logprob_all=np.concatenate((logprob1,logprob2,logprob3),axis=1)
    logprob_all_flipped=np.transpose(logprob_all)
    logprob_all_flipped=logprob_all_flipped[:,cut:]
    logprob_all_reshaped=logprob_all_flipped.reshape(-1)
    return logprob_all_reshaped

def chi2_pte(data,model,cov):
    inv_cov=np.linalg.inv(cov)
    min_chi2=np.dot(np.transpose((data-model)),  np.dot(inv_cov,(data-model)))

    mc=stats.multivariate_normal.rvs(cov=cov,size=10000)
    chi2_mc=np.array([np.dot(d, np.dot(inv_cov, d)) for d in mc])
    pte=(chi2_mc > min_chi2).sum()/chi2_mc.size

    return min_chi2,pte

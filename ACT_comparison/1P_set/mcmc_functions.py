import numpy as np
from astropy import units as u
import sys
sys.path.append('/home/cemoser/Projection_Codes/Mop-c-GT-copy/mopc')
import mopc_fft as mop
import ostrich.emulate
import ostrich.interpolate
from scipy import stats

#these are functions for building the 4d emulator for CMASS profiles and the accompanying mcmc functions


#----------------projection
home_mopc='/home/cemoser/Projection_Codes/Mop-c-GT-copy/'
beam_150_file = np.genfromtxt(home_mopc+'data/beam_f150_daynight.txt')
ell = beam_150_file[:,0]
beam_150_ell = beam_150_file[:,1]
ell2 = np.genfromtxt(home_mopc+'data/act_planck_s08_s18_cmb_f150_daynight_response_tsz.txt')[:,0]
res_150 = np.genfromtxt(home_mopc+'data/act_planck_s08_s18_cmb_f150_daynight_response_tsz.txt')[:,1]
nu=150.
z=0.54
sr2sqarcmin = 3282.8 * 60.**2

def fBeamF_150(x):
    return np.interp(x,ell,beam_150_ell)
def respT_150(x):
    return np.interp(x,ell2,res_150)

def project_profiles(prof,theta,z,nu,beam,respT,x,profile):
    if prof=='rho_mean':
        proj=mop.make_a_obs_profile_rho_array_ACT(theta,z,beam,x,profile)
    elif prof=='pth_mean':
        proj=mop.make_a_obs_profile_pth_array_ACT(theta,z,nu,beam,respT,x,profile)
    return proj


#----------emulator
usecols_w_dict={'rho_mean':(0,1),'pth_mean':(0,5)}
ASN1=np.array([0.25,0.32988,0.43528,0.57435,0.75786,1.0,1.31951,1.74110,2.29740,3.03143,4.0])
ASN2=np.array([0.5,0.57435,0.65975,0.75786,0.87055,1.14870,1.31951,1.51572,1.74110,2.0]) #take out 1.0                                                                                    
AAGN1=list(ASN1)
AAGN2=ASN2
AAGN1.pop(5)
AAGN1=np.array(AAGN1)
nums=np.linspace(22,65,44,dtype='int')
nums=list(nums)
nums.remove(38)
nums.remove(49)
nums.remove(60)
sims=['1P_'+str(n) for n in nums]
snap='024'

def samples_4d(ASN1,AAGN1,ASN2,AAGN2):
    samples=[]
    for asn1 in ASN1:
        samples.append([asn1,1.0,1.0,1.0])
    for agn1 in AAGN1:
        samples.append([1.0,agn1,1.0,1.0])
    for asn2 in ASN2:
        samples.append([1.0,1.0,asn2,1.0])
    for agn2 in AAGN2:
        samples.append([1.0,1.0,1.0,agn2])
    return np.array(samples)

def build_emulator(home,suite,prof):
    '''
    Be careful- this gives the profiles in log cgs!
    '''
    usecols=usecols_w_dict[prof]
    x,y=load_profiles(usecols,home,suite,sims,snap,prof)
    y=np.transpose(y)

    samples_emu=samples_4d(ASN1,AAGN1,ASN2,AAGN2)
    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples_emu,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':'linear'},
        num_components=12)
    return x,y,emulator

def load_profiles(usecols,home,suite,sims,snap,prof):
    #print("sims from load profiles",np.shape(sims),sims)
    y=[]
    for s in np.arange(len(sims)):
        f=home+suite+'/'+suite+'_'+sims[s]+'_'+snap+'_w.txt'
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
rescaling_factor_dict={'rho_mean':np.array([1.29434376,1.50186486,1.68729486,1.7872668,1.82233295,1.80844895,1.79453689,1.79890591,1.7996796]),'pth_mean':np.array([2.92388319,3.51593371,4.12355484,4.18725347,3.60379925,2.91902071,2.45097692,2.1838903,2.03076755])}

def add_dust(tsz_profile): #tsz in muK*sr
    #dust_profile=np.array([0.06473579,0.67299127,1.17076245,1.42560529,1.39368623,1.05834028,0.47767946,-0.16870933,-0.69811903]) #units muK*sqarcmin, this is what I got from subtracting my profiles
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
    ASN1,AAGN1,ASN2,AAGN2=theta
    y_proj=proj

    params=[[ASN1,AAGN1,ASN2,AAGN2]]
    profile=emulator(params).reshape(len(x))
    profile=10**profile #undo the log

    model_proj=project_profiles(prof,thta_arc,z,nu,fBeamF_150,respT_150,x,profile)
    #model_proj*=sr2sqarcmin
    
    #need to decide which to do first- rescaling or dust
    rescaling_factor=rescaling_factor_dict[prof]
    model_proj*=rescaling_factor

    if prof=='pth_mean': 
        #add dust correction
        model_proj=add_dust(model_proj)

    #rescaling
    #rescaling_factor=rescaling_factor_dict[prof]
    #model_proj*=rescaling_factor

    diff=y_proj-model_proj
    var= -0.5 * np.dot(diff, np.linalg.solve(cov, diff))
    return var

def lnprior(theta):
    ASN1,AAGN1,ASN2,AAGN2=theta
    if 0.25 < ASN1 < 4.0 and 0.25 < AAGN1 < 4.0 and 0.5 < ASN2 < 2.0 and 0.5 < AAGN2 < 2.0:
        return 0.0
    return -np.inf

#------- mcmc helper functions
load_data_dict={'rho_mean':'ksz','pth_mean':'tsz'}
def load_ACT_data(home,prof):
    sr2sqarcmin = 3282.8 * 60.**2
    file_string=load_data_dict[prof]
    thta_act,act_data,act_err=np.loadtxt(home+'diskring_'+file_string+'_varweight_measured.txt',usecols=(0,1,2),unpack=True)
    #act_data*=sr2sqarcmin
    #act_err*=sr2sqarcmin

    cov=np.loadtxt(home+'cov_diskring_'+file_string+'_varweight_bootstrap.txt')
    cov_diag=np.sqrt(np.diag(cov))
    #cov*=sr2sqarcmin
    #cov_diag*=sr2sqarcmin
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
    #inv_cov*=sr2sqarcmin
    min_chi2=np.dot(np.transpose((data-model)),  np.dot(inv_cov,(data-model)))

    mc=stats.multivariate_normal.rvs(cov=cov,size=10000)
    chi2_mc=np.array([np.dot(d, np.dot(inv_cov, d)) for d in mc])
    pte=(chi2_mc > min_chi2).sum()/chi2_mc.size

    return min_chi2,pte

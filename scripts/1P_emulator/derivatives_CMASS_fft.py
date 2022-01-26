import numpy as np
import time
import helper_functions_1P as fs
import sys
sys.path.append('/home/cemoser/Projection_Codes/Mop-c-GT-copy/mopc')
import mopc_fft as mop
import matplotlib.pyplot as plt
from matplotlib import gridspec

home_mop='/home/cemoser/Repositories/emu_CAMELS/mopc_profiles/'
home_mopc='/home/cemoser/Projection_Codes/Mop-c-GT-copy/'
suite=sys.argv[1]
prof=sys.argv[2]
func_str='linear'

snap='024'
z=fs.choose_redshift(suite)
Z_deriv=z[-1]

vary_arr=['ASN1','AAGN1','ASN2','AAGN2']
delt_theta={'ASN1':0.1,'AAGN1':0.1,'ASN2':0.05,'AAGN2':0.05}
#fiducial theta value
theta0=1.0
A_idx_theta0=5


#------------------------------------
#projection codes
beam_150_file = np.genfromtxt(home_mopc+'data/beam_f150_daynight.txt')
ell = beam_150_file[:,0]
beam_150_ell = beam_150_file[:,1]
ell2 = np.genfromtxt(home_mopc+'data/act_planck_s08_s18_cmb_f150_daynight_response_tsz.txt')[:,0]
res_150 = np.genfromtxt(home_mopc+'data/act_planck_s08_s18_cmb_f150_daynight_response_tsz.txt')[:,1]
nu=150.
theta_arc=np.linspace(0.7, 5., 6)

def fBeamF_150(x):
    return np.interp(x,ell,beam_150_ell) 
def respT_150(x):
    return np.interp(x,ell2,res_150)


def project_profiles(prof,theta,z,nu,beam,respT,x,profile):
    if prof=='rho_mean':
        proj=mop.make_a_obs_profile_rho_array(theta,z,beam,x,profile)
    elif prof=='pth_mean':
        proj=mop.make_a_obs_profile_pth_array(theta,z,nu,beam,respT,x,profile)
    return proj

#----------------------------------

start=time.time()
derivatives=[]
for count,val in enumerate(vary_arr):
    vary_str=val
    delta_theta=delt_theta[vary_str]
 
    vary,sims=fs.choose_vary(vary_str)
    sim_fiducial=sims[A_idx_theta0]
    samples,x,y,emulator=fs.build_emulator_CMASS(home_mop,suite,vary_str,prof,func_str)
    usecols=fs.usecols_w_dict[prof]
    
    x,fidu_profile=np.loadtxt(home_mop+suite+'/'+suite+'_'+sim_fiducial+'_'+snap+'_w.txt',usecols=usecols,unpack=True)
    fidu_profile=fs.cgs_units(prof,fidu_profile)

    #3d derivative plot
    yf=fidu_profile/fidu_profile
    #since our emulator is now weighted, we can reuse the uw function to produce +- profiles
    profile_plus,profile_minus=fs.compute_pm_profiles_CMASS(theta0,delta_theta,emulator,x)
    yp,ym=10**profile_plus/fidu_profile,10**profile_minus/fidu_profile
    yd=fs.derivative(profile_plus,profile_minus,delta_theta)
    ylabel=fs.choose_ylabel(prof,3)
    title=suite+' '+vary_str
    fs.plot_derivatives(x,yf,yp,ym,yd,ylabel,title,3)
    plt.savefig('/home/cemoser/Repositories/emu_CAMELS/figures/derivatives/CMASS_emulator/'+suite+'_'+vary_str+'_'+prof+'_deriv3d_CMASS.png')
    plt.close()

    proj_fidu=project_profiles(prof,theta_arc,Z_deriv,nu,fBeamF_150,respT_150,x,fidu_profile)
    proj_plus=project_profiles(prof,theta_arc,Z_deriv,nu,fBeamF_150,respT_150,x,10**profile_plus)
    proj_minus=project_profiles(prof,theta_arc,Z_deriv,nu,fBeamF_150,respT_150,x,10**profile_minus)
    proj_d=fs.derivative(proj_plus,proj_minus,delta_theta)
    ylabel=fs.choose_ylabel(prof,2)
    fs.plot_derivatives(theta_arc,proj_fidu,proj_plus,proj_minus,proj_d,ylabel,title,2)
    plt.savefig('/home/cemoser/Repositories/emu_CAMELS/figures/derivatives/CMASS_emulator/'+suite+'_'+vary_str+'_'+prof+'_deriv2d_fft_CMASS.png')
    plt.close()

    derivatives.append(proj_d)

derivatives=np.array(derivatives)
end=time.time()
print("it took %.2f minutes to create derivatives array"%((end-start)/60.))

np.savetxt('/home/cemoser/Repositories/emu_CAMELS/derivative_arrays/'+suite+'_'+prof+'_fft_CMASS.txt',derivatives)

import numpy as np
import time
import helper_functions_LH as fs
import sys
sys.path.append('/home/cemoser/Projection_Codes/Mop-c-GT-copy/mopc')
import mopc_gaussbeam as mop
import matplotlib.pyplot as plt
from matplotlib import gridspec

home_deriv = '/home/cemoser/Repositories/emu_CAMELS/derivative_arrays/'
home_mop='/home/cemoser/Repositories/emu_CAMELS/mopc_profiles/'
home_mopc='/home/cemoser/Projection_Codes/Mop-c-GT-copy/'
suite=sys.argv[1]
prof=sys.argv[2]
func_str='linear'

z=fs.choose_redshift(suite)
Z_deriv=z[-1]
omega_m = 0.3
sigma_8 = 0.8
M = 12.25

vary_arr=['ASN1','AAGN1','ASN2','AAGN2']
#same values as 1P, I tested that it doesn't affect the Fisher results
delt_theta={'ASN1':0.1,'AAGN1':0.1,'ASN2':0.05,'AAGN2':0.05}
#fiducial theta value
theta0=1.0

#------------------------------------
#projection codes
nu=150.
beam_gauss=1.4
theta_arc=np.linspace(0.7, 5., 6)

def project_profiles(prof,theta,z,nu,beam,x,profile):
    if prof=='rho_mean':
        proj=mop.make_a_obs_profile_rho_array(theta,z,beam,x,profile)
    elif prof=='pth_mean':
        proj=mop.make_a_obs_profile_pth_array(theta,z,nu,beam,x,profile)
    return proj

#----------------------------------

start=time.time()
derivatives=[]
for count,val in enumerate(vary_arr):
    vary_str=val
    delta_theta=delt_theta[vary_str]
 
    samples,x,y,emulator=fs.build_emulator_CMASS(home_mop,suite,prof,func_str)
    usecols=fs.usecols_w_dict[prof]
    
    fidu_params = [[omega_m,sigma_8,1.0,1.0,1.0,1.0,Z_deriv,M]]
    fidu_profile = emulator(fidu_params).reshape(len(x))

    profile_plus,profile_minus=fs.compute_pm_profiles_CMASS(vary_str,theta0,delta_theta,emulator,x,omega_m,sigma_8,Z_deriv,M)

    proj_fidu=project_profiles(prof,theta_arc,Z_deriv,nu,beam_gauss,x,10**fidu_profile)
    proj_plus=project_profiles(prof,theta_arc,Z_deriv,nu,beam_gauss,x,10**profile_plus)
    proj_minus=project_profiles(prof,theta_arc,Z_deriv,nu,beam_gauss,x,10**profile_minus)
    
    proj_d=fs.derivative(proj_plus,proj_minus,delta_theta)
    ylabel=fs.choose_ylabel(prof,2)
    title=suite+' '+fs.A_param_latex_dict[vary_str]
    fs.plot_derivatives(theta_arc,proj_fidu,proj_plus,proj_minus,proj_d,ylabel,title,2)
    plt.savefig(suite+'_'+vary_str+'_'+prof+'_deriv2d_gauss_CMASS.png')
    plt.close()

    derivatives.append(proj_d)

derivatives=np.array(derivatives)
end=time.time()
print("it took %.2f minutes to create derivatives array"%((end-start)/60.))

np.savetxt(home_deriv+suite+'_'+prof+'_gauss_CMASS.txt',derivatives)

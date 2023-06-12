import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import ostrich.emulate
import ostrich.interpolate
import mcmc_functions

z=0.54
nu=150.
#suite='SIMBA'
func_str='linear'

home_mop='/home/cemoser/Repositories/emu_CAMELS/mopc_profiles/'
home_ACT_data='/home/cemoser/Repositories/emu_CAMELS/ACT_comparison/'

sr2sqarcmin = 3282.8 * 60.**2
theta_act,ksz_act,cov_ksz_act,cov_diag_ksz=mcmc_functions.load_ACT_data(home_ACT_data,'rho_mean')
theta_act,tsz_act,cov_tsz_act,cov_diag_tsz=mcmc_functions.load_ACT_data(home_ACT_data,'pth_mean')
ksz_act_err=np.sqrt(np.diag(cov_ksz_act))
tsz_act_err=np.sqrt(np.diag(cov_tsz_act))
ksz_act_err,tsz_act_err=ksz_act_err*sr2sqarcmin,tsz_act_err*sr2sqarcmin
tsz_act*=sr2sqarcmin
ksz_act*=sr2sqarcmin


ksz_TNG,tsz_TNG=np.loadtxt(home_ACT_data+'proj_fft_original_TNG_w_muKsqarcmin_ACT_rbins.txt',usecols=(1,2),unpack=True)
tht_stef,tsz_TNG_dust=np.loadtxt(home_ACT_data+'fig6_TNG_H_dust.txt',usecols=(0,1),unpack=True)


x,y_rho_tng,emulator_rho_tng=mcmc_functions.build_emulator(home_mop,'IllustrisTNG','rho_mean')
x,y_pth_tng,emulator_pth_tng=mcmc_functions.build_emulator(home_mop,'IllustrisTNG','pth_mean')
x,y_rho_sim,emulator_rho_sim=mcmc_functions.build_emulator(home_mop,'SIMBA','rho_mean')
x,y_pth_sim,emulator_pth_sim=mcmc_functions.build_emulator(home_mop,'SIMBA','pth_mean')
rescale_factor_ksz=mcmc_functions.rescaling_factor_dict['rho_mean']
rescale_factor_tsz=mcmc_functions.rescaling_factor_dict['pth_mean']


sn1_ksz_tng,gn1_ksz_tng,sn2_ksz_tng,gn2_ksz_tng=0.837,1.128,1.737,1.019
sn1_tsz_tng,gn1_tsz_tng,sn2_tsz_tng,gn2_tsz_tng=0.25,1.0,1.0,1.0
sn1_ksz_sim,gn1_ksz_sim,sn2_ksz_sim,gn2_ksz_sim=0.69,0.25, 2.0, 0.5
sn1_tsz_sim,gn1_tsz_sim,sn2_tsz_sim,gn2_tsz_sim=1.0,1.0,1.0,1.74


params_ksz_tng=[[sn1_ksz_tng,gn1_ksz_tng,sn2_ksz_tng,gn2_ksz_tng]]
params_tsz_tng=[[sn1_tsz_tng,gn1_tsz_tng,sn2_tsz_tng,gn2_tsz_tng]]
params_ksz_sim=[[sn1_ksz_sim,gn1_ksz_sim,sn2_ksz_sim,gn2_ksz_sim]]
params_tsz_sim=[[sn1_tsz_sim,gn1_tsz_sim,sn2_tsz_sim,gn2_tsz_sim]]

prediction_rho_tng=emulator_rho_tng(params_ksz_tng)
prediction_rho_tng=prediction_rho_tng.reshape(len(x))
prediction_pth_tng=emulator_pth_tng(params_tsz_tng)
prediction_pth_tng=prediction_pth_tng.reshape(len(x))
prediction_rho_tng=10**prediction_rho_tng
prediction_pth_tng=10**prediction_pth_tng

prediction_rho_sim=emulator_rho_sim(params_ksz_sim)
prediction_rho_sim=prediction_rho_sim.reshape(len(x))
prediction_pth_sim=emulator_pth_sim(params_tsz_sim)
prediction_pth_sim=prediction_pth_sim.reshape(len(x))
prediction_rho_sim=10**prediction_rho_sim
prediction_pth_sim=10**prediction_pth_sim

#project
prediction_ksz_tng=mcmc_functions.project_profiles('rho_mean',theta_act,z,nu,mcmc_functions.fBeamF_150,mcmc_functions.respT_150,x,prediction_rho_tng)
prediction_tsz_tng=mcmc_functions.project_profiles('pth_mean',theta_act,z,nu,mcmc_functions.fBeamF_150,mcmc_functions.respT_150,x,prediction_pth_tng)
prediction_ksz_tng*=sr2sqarcmin
prediction_tsz_tng*=sr2sqarcmin
prediction_ksz_sim=mcmc_functions.project_profiles('rho_mean',theta_act,z,nu,mcmc_functions.fBeamF_150,mcmc_functions.respT_150,x,prediction_rho_sim)
prediction_tsz_sim=mcmc_functions.project_profiles('pth_mean',theta_act,z,nu,mcmc_functions.fBeamF_150,mcmc_functions.respT_150,x,prediction_pth_sim)
prediction_ksz_sim*=sr2sqarcmin
prediction_tsz_sim*=sr2sqarcmin

#rescale
prediction_ksz_rescaled_tng=prediction_ksz_tng*rescale_factor_ksz
prediction_tsz_rescaled_tng=prediction_tsz_tng*rescale_factor_tsz
prediction_ksz_rescaled_sim=prediction_ksz_sim*rescale_factor_ksz
prediction_tsz_rescaled_sim=prediction_tsz_sim*rescale_factor_tsz

#and add dust for tsz
prediction_tsz_rescaled_dust_tng=mcmc_functions.add_dust(prediction_tsz_rescaled_tng/sr2sqarcmin)
prediction_tsz_rescaled_dust_tng*=sr2sqarcmin
prediction_tsz_rescaled_dust_sim=mcmc_functions.add_dust(prediction_tsz_rescaled_sim/sr2sqarcmin)
prediction_tsz_rescaled_dust_sim*=sr2sqarcmin

#calculate chi2- the profiles and cov need to be in muK*sr 
min_chi2_ksz_tng,pte_ksz_tng=mcmc_functions.chi2_pte(ksz_act/sr2sqarcmin,prediction_ksz_rescaled_tng/sr2sqarcmin,cov_ksz_act)
min_chi2_tsz_tng,pte_tsz_tng=mcmc_functions.chi2_pte(tsz_act/sr2sqarcmin,prediction_tsz_rescaled_dust_tng/sr2sqarcmin,cov_tsz_act)
min_chi2_ksz_sim,pte_ksz_sim=mcmc_functions.chi2_pte(ksz_act/sr2sqarcmin,prediction_ksz_rescaled_sim/sr2sqarcmin,cov_ksz_act)
min_chi2_tsz_sim,pte_tsz_sim=mcmc_functions.chi2_pte(tsz_act/sr2sqarcmin,prediction_tsz_rescaled_dust_sim/sr2sqarcmin,cov_tsz_act)
min_chi2_TNG_ksz,pte_TNG_ksz=mcmc_functions.chi2_pte(ksz_act/sr2sqarcmin,ksz_TNG/sr2sqarcmin,cov_ksz_act)
min_chi2_TNG_tsz,pte_TNG_tsz=mcmc_functions.chi2_pte(tsz_act/sr2sqarcmin,tsz_TNG/sr2sqarcmin,cov_tsz_act)


#print(min_chi2_tsz)

fig,axes=plt.subplots(1,2,figsize=(12,5))
ax1=axes[0]
ax2=axes[1]

ax1.errorbar(theta_act,ksz_act,yerr=ksz_act_err,color='k',label='ACT (150 GHz)')
ax1.plot(theta_act,ksz_TNG,color='r',lw=2,label='original TNG, $\chi^2=$%.2f'%min_chi2_TNG_ksz)
ax1.plot(theta_act,prediction_ksz_rescaled_tng,color='b',linestyle='dashed',label='TNG emulator, $\chi^2=$%.2f'%min_chi2_ksz_tng)
ax1.plot(theta_act,prediction_ksz_rescaled_sim,color='purple',linestyle='dashed',label='SIMBA emulator, $\chi^2=$%.2f'%min_chi2_ksz_sim)

ax2.errorbar(theta_act,tsz_act,yerr=tsz_act_err,color='k')
ax2.plot(theta_act,tsz_TNG_dust,color='r',lw=2,label='original TNG, $\chi^2=$%.2f'%min_chi2_TNG_tsz)
ax2.plot(theta_act,prediction_tsz_rescaled_dust_tng,color='b',linestyle='dashed',label='TNG emulator, $\chi^2=$%.2f'%min_chi2_tsz_tng)
ax2.plot(theta_act,prediction_tsz_rescaled_dust_sim,color='purple',linestyle='dashed',label='SIMBA emulator, $\chi^2=$%.2f'%min_chi2_tsz_sim)

ax1.set_xlabel(r'$\theta$ (arcmin)',size=12)
ax2.set_xlabel(r'$\theta$ (arcmin)',size=12)
ax1.set_ylabel(r'$T_{kSZ}$',size=14)
ax2.set_ylabel(r'$T_{tSZ}$',size=14)

ax1.legend()
ax2.legend()
plt.savefig('emu4d_match_ACT.png',bbox_inches='tight')
plt.close()

np.savetxt('emu4d_match_ACT_profiles.txt',np.c_[theta_act,ksz_act,ksz_act_err,ksz_TNG,prediction_ksz_rescaled_tng,prediction_ksz_rescaled_sim,tsz_act,tsz_act_err,tsz_TNG_dust,prediction_tsz_rescaled_dust_tng,prediction_tsz_rescaled_dust_sim],header='theta (arcmin), kSZ ACT, kSZ ACT err, kSZ original TNG, TNG emu kSZ best, SIM emu kSZ best, tSZ ACT, tSZ ACT err, tSZ original TNG (with dust), TNG emu tSZ best, SIM emu tSZ best \n Best params for TNG emu kSZ: [%.2f, %.2f, %.2f, %.2f], chi2: %.2f \n Best params for SIM emu kSZ: [%.2f, %.2f, %.2f, %.2f], chi2: %.2f \n Best params for TNG emu tSZ: [%.2f, %.2f, %.2f, %.2f], chi2: %.2f \n Best params for SIM emu tSZ: [%.2f, %.2f, %.2f, %.2f], chi2: %.2f'%(sn1_ksz_tng,gn1_ksz_tng,sn2_ksz_tng,gn2_ksz_tng,min_chi2_ksz_tng,sn1_ksz_sim,gn1_ksz_sim,sn2_ksz_sim,gn2_ksz_sim,min_chi2_ksz_sim,sn1_tsz_tng,gn1_tsz_tng,sn2_tsz_tng,gn2_tsz_tng,min_chi2_tsz_tng,sn1_tsz_sim,gn1_tsz_sim,sn2_tsz_sim,gn2_tsz_sim,min_chi2_tsz_sim))

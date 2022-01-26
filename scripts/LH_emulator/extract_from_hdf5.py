import matplotlib.pyplot as plt
import numpy             as np
import profile_functions
import h5py
import sys

home='/home/cemoser/Repositories/emu_CAMELS/emulator_profiles/LH_set' 

#-----------------------------------input section
suite=sys.argv[1]
#suite='IllustrisTNG'
emulator_type='general'
  
nums=np.linspace(0,999,1000,dtype='int')
simulations=['LH_'+str(n) for n in nums]

if emulator_type=='general':
    #snap_arr=['033','032','031','030','029','028','027','026','025','024']
    snap_arr=['024'] #single redshift for now

    #higher mass bins giving me trouble, just do lower 2 for now
    mass_str_arr=['11-11.5','11.5-12']#,'12-12.3','12.3-13.1']
    mh_low_arr=[10**11.,10**11.5+0.1]#,10**12.+0.1]#,10**12.3+0.1]
    mh_high_arr=[10**11.5,10**12.]#,10**12.3]#,10**13.1]
    mh_low_pow_arr=[11,11.5]#,12]#,12.3]
    mh_high_pow_arr=[11.5,12]#,12.3]#,13.1]

elif emulator_type=='CMASS':
    snap_arr=['024']
    mass_str_arr=['12.12-13.2'] #13.98, there are no halos above 13.1
    mh_low_arr=[10**12.12]
    mh_high_arr=[10**13.2] #13.98
    mh_low_pow_arr=[12.12]
    mh_high_pow_arr=[13.2]

#--------------------------------------------------------------- 
if suite=='SIMBA':
    z=0.53761
elif suite=='IllustrisTNG':
    z=0.53726

def extract(simulation,snap):
    #change path to hdf5 files as necessary
    stacks=h5py.File(home+'/hdf5_files/'+suite+'_'+simulation+'_'+snap+'.hdf5','r')
    val            = stacks['Profiles']
    val_dens       = np.array(val[0,:,:])
    val_pres       = np.array(val[1,:,:])
    val_metal_gmw  = np.array(val[2,:,:])
    val_temp_gmw   = np.array(val[3,:,:])
    bins           = np.array(stacks['nbins'])
    r              = np.array(stacks['r'])
    nprofs         = np.array(stacks['nprofs'])
    mh             = np.array(stacks['Group_M_Crit200'])
    rh             = np.array(stacks['Group_R_Crit200'])
    GroupFirstSub  = np.array(stacks['GroupFirstSub'])
    sfr            = np.array(stacks['GroupSFR']) 
    mstar          = np.array(stacks['GroupMassType_Stellar']) 
    return z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw

for j in np.arange(len(simulations)):
    sim=simulations[j]
    for k in np.arange(len(snap_arr)):
        snap=snap_arr[k]
        z,val_dens,bins,r,val_pres,nprofs,mh,rh,GroupFirstSub,sfr,mstar,val_metal_gmw,val_temp_gmw=extract(sim,snap)
        h=0.6711
            
        mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw=profile_functions.correct(z,h,mh,mstar,rh,val_dens,val_pres,r,val_temp_gmw)
    
        for m in np.arange(len(mh_low_arr)):
            mh_low=mh_low_arr[m]
            mh_high=mh_high_arr[m]
            mass_str=mass_str_arr[m]
            mh_low_pow=mh_low_pow_arr[m]
            mh_high_pow=mh_high_pow_arr[m]
            #print(sim,snap,mass_str)
            mstarm,mhm,rhm,sfrm,GroupFirstSubm,val_presm,val_densm,nprofsm,val_metal_gmwm,val_temp_gmwm=profile_functions.mhalo_cut(mh_low,mh_high,mstar,mh,rh,sfr,GroupFirstSub,val_pres,val_dens,val_metal_gmw,val_temp_gmw,bins)
            print(sim,mass_str,nprofsm)
    
            r_mpc=r/1.e3
            print(r_mpc)
            #outer cut 20, inner cut 5e-4 usually. LH set requires stricter outer cuts:
            #outer cut 2 for TNG, 
            r_mpc_cut,val_densm=profile_functions.outer_cut_multi(1,r_mpc,val_densm)
            r_mpc_cut2,val_densm=profile_functions.inner_cut_multi(5.e-4,r_mpc_cut,val_densm)
            r_mpc_cut,val_presm=profile_functions.outer_cut_multi(1,r_mpc,val_presm)
            r_mpc_cut2,val_presm=profile_functions.inner_cut_multi(5.e-4,r_mpc_cut,val_presm)
            r_mpc_cut,val_metal_gmwm=profile_functions.outer_cut_multi(1,r_mpc,val_metal_gmwm)
            r_mpc_cut2,val_metal_gmwm=profile_functions.inner_cut_multi(5.e-4,r_mpc_cut,val_metal_gmwm)
            r_mpc_cut,val_temp_gmwm=profile_functions.outer_cut_multi(1,r_mpc,val_temp_gmwm)
            r_mpc_cut2,val_temp_gmwm=profile_functions.inner_cut_multi(5.e-4,r_mpc_cut,val_temp_gmwm)
                        
            mean_unnorm_densm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_densm)
            mean_unnorm_presm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_presm)
            median_unnorm_densm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_densm)
            median_unnorm_presm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_presm)
            mean_unnorm_metal_gmwm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_metal_gmwm)
            mean_unnorm_temp_gmwm=np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]),0,val_temp_gmwm)
            median_unnorm_metal_gmwm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_metal_gmwm)
            median_unnorm_temp_gmwm=np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]),0,val_temp_gmwm)
        
            errup_dens_unnormm,errlow_dens_unnormm,std_dens_unnormm=profile_functions.get_errors(val_densm)
            errup_pres_unnormm,errlow_pres_unnormm,std_pres_unnormm=profile_functions.get_errors(val_presm)
            errup_metal_gmw_unnormm,errlow_metal_gmw_unnormm,std_metal_gmw_unnormm=profile_functions.get_errors(val_metal_gmwm)
            errup_temp_gmw_unnormm,errlow_temp_gmw_unnormm,std_temp_gmw_unnormm=profile_functions.get_errors(val_temp_gmwm)
        
        
            if emulator_type=='general':
                header='R (Mpc), mean rho (Msol/kpc^3), errup (Msol/kpc^3), errlow, std, median rho (Msol/kpc^3), mean pth (Msol/kpc/s^2), errup(Msol/kpc/s^2), errlow, std, median pth (Msol/kpc/s^2), mean gas-mass-weighted metal (fraction), errup, errlow, std, median metal, mean gas-mass-weighted temp (K), errup, errlow, std, median temp  \n nprofs %i, mean mh %f, median mh %f \n Mass range %.2f - %.1f'%(nprofsm,np.mean(mhm),np.median(mhm),mh_low_pow,mh_high_pow)
                np.savetxt(home+'/'+suite+'/'+suite+'_'+sim+'_'+snap+'_uw_%s.txt'%mass_str,np.c_[r_mpc_cut2,mean_unnorm_densm, errup_dens_unnormm,errlow_dens_unnormm,std_dens_unnormm,median_unnorm_densm,mean_unnorm_presm,errup_pres_unnormm,errlow_pres_unnormm,std_pres_unnormm,median_unnorm_presm,mean_unnorm_metal_gmwm, errup_metal_gmw_unnormm,errlow_metal_gmw_unnormm,std_metal_gmw_unnormm,median_unnorm_metal_gmwm,mean_unnorm_temp_gmwm, errup_temp_gmw_unnormm,errlow_temp_gmw_unnormm,std_temp_gmw_unnormm,median_unnorm_temp_gmwm],header=header)
        
            if emulator_type=='CMASS':
                mean_mh,mean_unnorm_dens_w,mean_unnorm_pres_w=profile_functions.mass_distribution_weight(mhm,val_densm,val_presm)
                header='R (Mpc), rho (Msol/kpc^3), errup, errlow, std, pth (Msol/kpc/s^2), errup, errlow, std \n nprofs %i, average weighted mh %f'%(nprofs,mean_mh)
                mean_masses_w[sim]=mean_mh
                np.savetxt(home+'/'+suite+'_'+sim+'_'+snap+'_w.txt',np.c_[r_mpc_cut2,mean_unnorm_dens_w, errup_dens_unnormm,errlow_dens_unnormm,std_dens_unnormm,mean_unnorm_pres_w,errup_pres_unnormm,errlow_pres_unnormm,std_pres_unnormm],header=header)

    

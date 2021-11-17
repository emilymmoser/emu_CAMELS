import matplotlib.pyplot as plt
import numpy as np
import ostrich.interpolate
import ostrich.emulate
import helper_functions as fs
import sys
import time


home='/home/cemoser/Repositories/emu_CAMELS/emulator_profiles/'
suite=sys.argv[1]
vary_str=sys.argv[2]
prof=sys.argv[3]
func_str='linear'

print("---------------------------")
print(suite,prof,"feedback",vary_str)
print("---------------------------")

mass=fs.mass
mass_str=fs.mass_str
snap=fs.snap
z=fs.choose_redshift(suite)
vary,sims=fs.choose_vary(vary_str)
samples=fs.cartesian_prod(vary,z,mass) 
nsamp=samples.shape[0]

usecols=fs.usecols_dict[prof]

samples,x,y,emulator=fs.build_emulator_3D(home,suite,vary_str,prof,func_str)
'''
start=time.time()
errs_drop_1,emulated_drop_1=fs.drop1_test(y,nsamp,samples)
end=time.time()

print("drop 1 test took %.2f minutes, %.2f hours"%((end-start)/60.,(end-start)/3600.))
np.savetxt('/home/cemoser/Repositories/emu_CAMELS/tests/radial_errs_general_emu/errs_drop1_'+suite+'_'+vary_str+'_'+prof+'_4Mbins_Rbflin.txt',errs_drop_1)
np.savetxt('/home/cemoser/Repositories/emu_CAMELS/tests/radial_errs_general_emu/emulated_drop1_'+suite+'_'+vary_str+'_'+prof+'_4Mbins_Rbflin.txt',emulated_drop_1) 
'''

errs_drop_1=np.genfromtxt('/home/cemoser/Repositories/emu_CAMELS/tests/radial_errs_general_emu/errs_drop1_'+suite+'_'+vary_str+'_'+prof+'_4Mbins_Rbflin.txt')
emulated_drop_1=np.genfromtxt('/home/cemoser/Repositories/emu_CAMELS/tests/radial_errs_general_emu/emulated_drop1_'+suite+'_'+vary_str+'_'+prof+'_4Mbins_Rbflin.txt')


m_idx,z_idx=2,0
A_idx=np.linspace(0,10,11,dtype='int')
y_filtered=[]
emulated_filtered=[]
errs_filtered=[]
for a in A_idx:
    index=fs.retrieve_index_3D(a,z_idx,m_idx)
    y_filtered.append(y[:,index])
    emulated_filtered.append(emulated_drop_1[:,index])
    errs_filtered.append(errs_drop_1[:,index])
y_filtered=np.array(y_filtered)
emulated_filtered=np.array(emulated_filtered)
errs_filtered=np.array(errs_filtered)

ylabel=fs.choose_ylabel(prof,3)
title='%s, M = %.2f, z = %.2f'%(suite,mass[m_idx],z[z_idx])
fig=fs.plot_drop1_percent_err(x,y_filtered,emulated_filtered,errs_filtered,vary,vary_str,ylabel,title)

plt.savefig('/home/cemoser/Repositories/emu_CAMELS/tests/radial_errs_general_emu/radial_errs_'+suite+'_'+prof+'_'+vary_str+'_z'+str(z_idx)+'_M'+str(m_idx)+'.png',bbox_inches='tight')

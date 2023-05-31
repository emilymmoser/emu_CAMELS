import matplotlib.pyplot as plt
import numpy as np
import ostrich.interpolate
import ostrich.emulate
import helper_functions as fs
import sys
import time


home='/home/cemoser/Repositories/emu_CAMELS/emulator_profiles/1P_set/'
suite=sys.argv[1]
vary_str=sys.argv[2]
prof=sys.argv[3]
func_str='linear'

def outer_cut_multi(outer_cut,x,arr):
    idx=np.where(x <= outer_cut)
    idx=np.array(idx[0])
    x,arr=x[idx],arr[:,idx]
    return x,arr


print("---------------------------")
print(suite,prof,"feedback",vary_str)
print("---------------------------")

mass=fs.mass
mass_str=fs.mass_str
mass_range_latex=fs.mass_range_latex
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

print("y",np.shape(y))
print("errs_drop1",np.shape(errs_drop_1))

m_idx,z_idx=2,9 #2,0 for the paper
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

print("y_filtered",np.shape(y_filtered))
print("errs_filtered",np.shape(errs_filtered))

x_cut,y_filtered=outer_cut_multi(10,x,y_filtered)
x_cut,emulated_filtered=outer_cut_multi(10,x,emulated_filtered)
x,errs_filtered=outer_cut_multi(10,x,errs_filtered)

ylabel=fs.choose_ylabel(prof,3)
title=r'%s, %s, z = %.2f'%(suite,mass_range_latex[m_idx],z[z_idx])
fig=fs.plot_drop1_percent_err(x,y_filtered,emulated_filtered,errs_filtered,vary,vary_str,ylabel,title)

plt.savefig('/home/cemoser/Repositories/emu_CAMELS/tests/radial_errs_general_emu/radial_errs_'+suite+'_'+prof+'_'+vary_str+'_z'+str(z_idx)+'_M'+str(m_idx)+'.png',bbox_inches='tight')

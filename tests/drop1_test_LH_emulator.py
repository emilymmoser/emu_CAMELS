import matplotlib.pyplot as plt
import numpy as np
import ostrich.interpolate
import ostrich.emulate
import helper_functions_LH as fs
import sys
import time


home='/home/cemoser/Repositories/emu_CAMELS/emulator_profiles/LH_set/'
suite=sys.argv[1]
prof=sys.argv[2]
func_str='linear'

print("---------------------------")
print("LH drop 1 test",suite,prof)
print("---------------------------")

mass=fs.mass
mass_str=fs.mass_str
mass_range_latex=fs.mass_range_latex
snap='024'
z=fs.choose_redshift(suite)
z=np.array([z[-1]])

omegam,sigma8,ASN1,AAGN1,ASN2,AAGN2=np.loadtxt('CosmoAstroSeed_%s.txt'%suite,usecols=(1,2,3,4,5,6),unpack=True)

params = np.vstack([omegam,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
samples=fs.LH_cartesian_prod(params,z,mass) 
nsamp=samples.shape[0]
usecols=fs.usecols_dict[prof]
samples,x,y,emulator=fs.build_emulator_3D(home,suite,prof,func_str)


#----------------------------
#drop1 test
start=time.time()
errs_drop_1,emulated_drop_1=fs.drop1_test(y,nsamp,samples)
end=time.time()

print("drop 1 test took %.2f minutes, %.2f hours"%((end-start)/60.,(end-start)/3600.))
np.savetxt('errs_drop1_'+suite+'_'+prof+'.txt',errs_drop_1)
np.savetxt('emulated_drop1_'+suite+'_'+prof+'.txt',emulated_drop_1) 
#----------------------

#if loading in separately
#errs_drop_1=np.genfromtxt('errs_drop1_'+suite+'_'+prof+'.txt')
#emulated_drop_1=np.genfromtxt('emulated_drop1_'+suite+'_'+prof+'.txt')


for i in np.arange(5):
    index=np.random.randint(0,1000)
    print("random index",index,omegam[index],sigma8[index],ASN1[index],AAGN1[index],ASN2[index],AAGN2[index])
    y_filtered=y[:,index]
    emulated_filtered=emulated_drop_1[:,index]
    errs_filtered=errs_drop_1[:,index]
    y_filtered=np.array(y_filtered)
    emulated_filtered=np.array(emulated_filtered)
    errs_filtered=np.array(errs_filtered)

    ylabel=fs.choose_ylabel(prof,3)
    title=r'%s, z = %.2f'%(suite,z)
    fig=fs.plot_drop1_percent_err_LH(x,y_filtered,emulated_filtered,errs_filtered,ylabel,title)

    plt.savefig('emu_errs_'+suite+'_'+prof+'_index_'+str(index)+'.png',bbox_inches='tight')

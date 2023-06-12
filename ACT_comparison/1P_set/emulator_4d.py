import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import ostrich.emulate
import ostrich.interpolate

z=0.54
suite='IllustrisTNG'
#prof='rho_mean'
func_str='linear'

home_mop='/home/cemoser/Repositories/emu_CAMELS/mopc_profiles/'


ASN1=np.array([0.25,0.32988,0.43528,0.57435,0.75786,1.0,1.31951,1.74110,2.29740,3.03143,4.0])
ASN2=np.array([0.5,0.57435,0.65975,0.75786,0.87055,1.14870,1.31951,1.51572,1.74110,2.0]) #take out 1.0

AAGN1=list(ASN1)
AAGN2=ASN2
#remove the degenerate samples
AAGN1.pop(5)
AAGN1=np.array(AAGN1)

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

samples=samples_4d(ASN1,AAGN1,ASN2,AAGN2)
#print("samples",samples)
nsamp=samples.shape[0]

'''
#check that the degenerate samples are the same. Result: they are the same 
#(1.0,1.0,1.0,1.0)=1P_27,38,49,60.  usecols= (0,1) for rho, (0,5) for pth

x,fidu_asn1=np.loadtxt(home_mop+suite+'/'+suite+'_1P_27_024_w.txt',usecols=(0,5),unpack=True)
x,fidu_agn1=np.loadtxt(home_mop+suite+'/'+suite+'_1P_38_024_w.txt',usecols=(0,5),unpack=True)
x,fidu_asn2=np.loadtxt(home_mop+suite+'/'+suite+'_1P_49_024_w.txt',usecols=(0,5),unpack=True)
x,fidu_agn2=np.loadtxt(home_mop+suite+'/'+suite+'_1P_60_024_w.txt',usecols=(0,5),unpack=True)

plt.semilogx(x,fidu_asn1/fidu_asn1,color='b',label='ASN1')
plt.semilogx(x,fidu_agn1/fidu_asn1,color='r',label='AAGN1')
plt.semilogx(x,fidu_asn2/fidu_asn1,color='g',label='ASN2')
plt.semilogx(x,fidu_agn2/fidu_asn1,color='purple',label='AAGN2')
plt.xlabel('R (Mpc)',size=12)
plt.ylabel(r'$\rho_{gas}$',size=12)
plt.legend()
plt.savefig('degenerate_samples_emu4d.png',bbox_inches='tight')
plt.close()
'''

nums=np.linspace(22,65,44,dtype='int')
nums=list(nums)
nums.remove(38)
nums.remove(49)
nums.remove(60)
sims=['1P_'+str(n) for n in nums]
snap='024'
usecols_w_dict={'rho_mean':(0,1),'pth_mean':(0,5)}


def build_emulator(home,suite,prof):
    usecols=usecols_w_dict[prof]
    x,y=load_profiles(usecols,home,suite,sims,snap,prof)
    y=np.transpose(y)

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=12)
    return x,y,emulator

def load_profiles(usecols,home,suite,sims,snap,prof):
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
        #input rho in Msol/kpc^3
        rho=arr*u.solMass/u.kpc/u.kpc/u.kpc
        arr=rho.cgs
    elif prof=='pth_mean':
        #input pth in Msol/kpc/s^2
        pth=arr*u.solMass/u.kpc/(u.s*u.s)
        arr=pth.to(u.dyne/(u.cm*u.cm))
    return arr.value


x,y_rho,emulator_rho=build_emulator(home_mop,suite,'rho_mean')
x,y_pth,emulator_pth=build_emulator(home_mop,suite,'pth_mean')


sn1,gn1,sn2,gn2=1.4,2.7,1.2,0.8
params=[[sn1,gn1,sn2,gn2]]
prediction_rho=emulator_rho(params)
prediction_rho=prediction_rho.reshape(len(x))
prediction_pth=emulator_pth(params)
prediction_pth=prediction_pth.reshape(len(x))


a,b,c,d=1.4,1.0,1.0,1.0
par=[[a,b,c,d]]
predic_rho=emulator_rho(par)
predic_rho=predic_rho.reshape(len(x))
predic_pth=emulator_pth(par)
predic_pth=predic_pth.reshape(len(x))


a2,b2,c2,d2=1.0,2.7,1.0,1.0
par2=[[a2,b2,c2,d2]]
predic2_rho=emulator_rho(par2)
predic2_rho=predic2_rho.reshape(len(x))
predic2_pth=emulator_pth(par2)
predic2_pth=predic2_pth.reshape(len(x))


a3,b3,c3,d3=1.0,1.0,1.2,1.0
par3=[[a3,b3,c3,d3]]
predic3_rho=emulator_rho(par3)
predic3_rho=predic3_rho.reshape(len(x))
predic3_pth=emulator_pth(par3)
predic3_pth=predic3_pth.reshape(len(x))


a4,b4,c4,d4=1.0,1.0,1.0,0.8
par4=[[a4,b4,c4,d4]]
predic4_rho=emulator_rho(par4)
predic4_rho=predic4_rho.reshape(len(x))
predic4_pth=emulator_pth(par4)
predic4_pth=predic4_pth.reshape(len(x))

#get the surrounding profiles to see if the prediction makes sense
fiducial_rho,fiducial_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_27_024_w.txt',usecols=(1,5),unpack=True)
asn1_low_rho,asn1_low_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_28_024_w.txt',usecols=(1,5),unpack=True)
asn1_high_rho,asn1_high_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_29_024_w.txt',usecols=(1,5),unpack=True)
agn1_low_rho,agn1_low_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_41_024_w.txt',usecols=(1,5),unpack=True)
agn1_high_rho,agn1_high_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_42_024_w.txt',usecols=(1,5),unpack=True)
asn2_low_rho,asn2_low_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_50_024_w.txt',usecols=(1,5),unpack=True)
asn2_high_rho,asn2_high_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_51_024_w.txt',usecols=(1,5),unpack=True)
agn2_low_rho,agn2_low_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_58_024_w.txt',usecols=(1,5),unpack=True)
agn2_high_rho,agn2_high_pth=np.loadtxt(home_mop+suite+'/'+suite+'_1P_59_024_w.txt',usecols=(1,5),unpack=True)

fiducial_rho,fiducial_pth=cgs_units('rho_mean',fiducial_rho),cgs_units('pth_mean',fiducial_pth)
asn1_low_rho,asn1_low_pth=cgs_units('rho_mean',asn1_low_rho),cgs_units('pth_mean',asn1_low_pth)
asn1_high_rho,asn1_high_pth=cgs_units('rho_mean',asn1_high_rho),cgs_units('pth_mean',asn1_high_pth)
agn1_low_rho,agn1_low_pth=cgs_units('rho_mean',agn1_low_rho),cgs_units('pth_mean',agn1_low_pth)
agn1_high_rho,agn1_high_pth=cgs_units('rho_mean',agn1_high_rho),cgs_units('pth_mean',agn1_high_pth)
asn2_low_rho,asn2_low_pth=cgs_units('rho_mean',asn2_low_rho),cgs_units('pth_mean',asn2_low_pth)
asn2_high_rho,asn2_high_pth=cgs_units('rho_mean',asn2_high_rho),cgs_units('pth_mean',asn2_high_pth)
agn2_low_rho,agn2_low_pth=cgs_units('rho_mean',agn2_low_rho),cgs_units('pth_mean',agn2_low_pth)
agn2_high_rho,agn2_high_pth=cgs_units('rho_mean',agn2_high_rho),cgs_units('pth_mean',agn2_high_pth)



fig=plt.figure(figsize=(16,12))

plt.subplot(2,2,1)
plt.semilogx(x,10**predic_rho/fiducial_rho,color='lawngreen',label='emu (%.1f-%.1f-%.1f-%.1f)'%(a,b,c,d))
plt.semilogx(x,10**predic2_rho/fiducial_rho,color='blueviolet',label='emu (%.1f-%.1f-%.1f-%.1f)'%(a2,b2,c2,d2))
plt.semilogx(x,10**predic3_rho/fiducial_rho,color='firebrick',label='emu (%.1f-%.1f-%.1f-%.1f)'%(a3,b3,c3,d3))
plt.semilogx(x,10**predic4_rho/fiducial_rho,color='dodgerblue',label='emu (%.1f-%.1f-%.1f-%.1f)'%(a4,b4,c4,d4))
plt.semilogx(x,fiducial_rho/fiducial_rho,color='cyan',label='fiducial (1-1-1-1)')
plt.semilogx(x,10**prediction_rho/fiducial_rho,color='k',label='emu (%.1f-%.1f-%.1f-%.1f)'%(sn1,gn1,sn2,gn2))
plt.xlabel('R (Mpc)',size=16)
plt.ylabel(r'$\frac{\rho}{\rho_0}$',size=20)
plt.title('Emulator Predictions',size=20)
plt.legend(fontsize=10,loc='upper right')


plt.subplot(2,2,2)
plt.semilogx(x,10**predic_pth/fiducial_pth,color='lawngreen',label='emu (%.1f-%.1f-%.1f-%.1f)'%(a,b,c,d))
plt.semilogx(x,10**predic2_pth/fiducial_pth,color='blueviolet',label='emu (%.1f-%.1f-%.1f-%.1f)'%(a2,b2,c2,d2))
plt.semilogx(x,10**predic3_pth/fiducial_pth,color='firebrick',label='emu (%.1f-%.1f-%.1f-%.1f)'%(a3,b3,c3,d3))
plt.semilogx(x,10**predic4_pth/fiducial_pth,color='dodgerblue',label='emu (%.1f-%.1f-%.1f-%.1f)'%(a4,b4,c4,d4))
plt.semilogx(x,fiducial_pth/fiducial_pth,color='cyan',label='fiducial (1-1-1-1)')
plt.semilogx(x,10**prediction_pth/fiducial_pth,color='k',label='emu (%.1f-%.1f-%.1f-%.1f)'%(sn1,gn1,sn2,gn2))
plt.xlabel('R (Mpc)',size=16)
plt.ylabel(r'$\frac{P}{P_0}$',size=20)
plt.title('Emulator Predictions',size=20)

plt.subplot(2,2,3)
plt.semilogx(x,asn1_low_rho/fiducial_rho,color='b',linestyle='dotted',label='ASN1=2.3')
plt.semilogx(x,asn1_high_rho/fiducial_rho,color='b',linestyle='dashed',label='ASN1=3.0')
plt.semilogx(x,agn1_low_rho/fiducial_rho,color='r',linestyle='dotted',label='AGN1=0.57')
plt.semilogx(x,agn1_high_rho/fiducial_rho,color='r',linestyle='dashed',label='AGN1=0.76')
plt.semilogx(x,asn2_low_rho/fiducial_rho,color='g',linestyle='dotted',label='ASN2=1.15')
plt.semilogx(x,asn2_high_rho/fiducial_rho,color='g',linestyle='dashed',label='ASN2=1.3')
plt.semilogx(x,agn2_low_rho/fiducial_rho,color='purple',linestyle='dotted',label='AGN2=0.76')
plt.semilogx(x,agn2_high_rho/fiducial_rho,color='purple',linestyle='dashed',label='AGN2=0.87')
plt.semilogx(x,fiducial_rho/fiducial_rho,color='cyan')
plt.semilogx(x,10**prediction_rho/fiducial_rho,color='k',label='emu (%.1f-%.1f-%.1f-%.1f)'%(sn1,gn1,sn2,gn2))
plt.xlabel('R (Mpc)',size=16)
plt.ylabel(r'$\frac{\rho}{\rho_0}$',size=20)
plt.title('Real Profiles',size=20)
plt.legend(fontsize=10,loc='upper right')

plt.subplot(2,2,4)
plt.semilogx(x,asn1_low_pth/fiducial_pth,color='b',linestyle='dotted',label='ASN1=2.3')
plt.semilogx(x,asn1_high_pth/fiducial_pth,color='b',linestyle='dashed',label='ASN1=3.0')
plt.semilogx(x,agn1_low_pth/fiducial_pth,color='r',linestyle='dotted',label='AGN1=0.57')
plt.semilogx(x,agn1_high_pth/fiducial_pth,color='r',linestyle='dashed',label='AGN1=0.76')
plt.semilogx(x,asn2_low_pth/fiducial_pth,color='g',linestyle='dotted',label='ASN2=1.15')
plt.semilogx(x,asn2_high_pth/fiducial_pth,color='g',linestyle='dashed',label='ASN2=1.3')
plt.semilogx(x,agn2_low_pth/fiducial_pth,color='purple',linestyle='dotted',label='AGN2=0.76')
plt.semilogx(x,agn2_high_pth/fiducial_pth,color='purple',linestyle='dashed',label='AGN2=0.87')
plt.semilogx(x,fiducial_rho/fiducial_rho,color='cyan')
plt.semilogx(x,10**prediction_pth/fiducial_pth,color='k',label='emu (%.1f-%.1f-%.1f-%.1f)'%(sn1,gn1,sn2,gn2))
plt.xlabel('R (Mpc)',size=16)
plt.ylabel(r'$\frac{P}{P_0}$',size=20)
plt.title('Real Profiles',size=20)

plt.savefig('prediction_emu4d_'+suite+'.png',bbox_inches='tight')

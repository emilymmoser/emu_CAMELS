import numpy as np
import helper_functions_LH as fs


home='/home/cemoser/Repositories/emu_CAMELS/emulator_profiles/LH_set/'
suite='IllustrisTNG' 
prof='rho_mean'
func_str='linear'

#just try single mass and redshift for now, 11-11.5 and z=0.5
mass=fs.mass
mass_str=fs.mass_str
snap='024'
z=fs.choose_redshift(suite)
z=z[-1]

omegam,sigma8,ASN1,AAGN1,ASN2,AAGN2=np.loadtxt('CosmoAstroSeed_%s.txt'%suite,usecols=(1,2,3,4,5,6),unpack=True)

samples,x,y,emulator=fs.build_emulator_3D(home,suite,prof,func_str,omegam,sigma8,ASN1,AAGN1,ASN2,AAGN2)


om=0.3
s8=0.8
asn1=0.5
agn1=1.2
asn2=0.7
agn2=1.5
params=[[om,s8,asn1,agn1,asn2,agn2]]
emulated=emulator(params)
print(np.shape(emulated))

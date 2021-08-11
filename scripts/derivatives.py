import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import scipy.interpolate
import numpy as np
import warnings
import random
import time
import ostrich.emulate
import ostrich.interpolate
import functions as fs
import sys


home='/home/cemoser/Repositories/ostrich/Emulator_profiles/'
suite=sys.argv[1]
vary_str=sys.argv[2]
prof=sys.argv[3]
func_str='linear'

mass=fs.mass
mass_str=fs.mass_str
snap=fs.snap
z=fs.choose_redshift(suite)
vary,sims=fs.choose_vary(vary_str)
samples=fs.cartesian_prod(vary,z,mass) 
nsamp=samples.shape[0]

samples,x,y,emulator=fs.build_emulator_3D(home,suite,vary_str,prof,func_str)

if prof=='rho_mean':
    ylabel=r'$\frac{\rho_{mean}}{\rho_0}$'
elif prof=='rho_med':
    ylabel=r'$\frac{\rho_{med}}{\rho_0}$'
elif prof=='pth_mean':
    ylabel=r'$\frac{P_{mean}}{P_0}$'
elif prof=='pth_med':
    ylabel=r'$\frac{P_{med}}{P_0}$'

if vary_str=='ASN1':
    vary_label=r'$A_{SN1}$'
    delta_theta=[0.01,0.1,0.2,0.3,0.5]
elif vary_str=='ASN2':
    vary_label=r'$A_{SN2}$'
    delta_theta=[0.01,0.05,0.1,0.15,0.25]
elif vary_str=='AAGN1':
    vary_label=r'$A_{AGN1}$'
    delta_theta=[0.01,0.1,0.2,0.3,0.5]
elif vary_str=='AAGN2':
    vary_label=r'$A_{AGN2}$'
    delta_theta=[0.01,0.05,0.1,0.15,0.25]


#filter the y vals back down to just the 11 profiles varying the feedback param
m_idx,z_idx=1,2
A_idx=np.linspace(0,10,11,dtype='int')
y_filtered=[]
for a in A_idx:
    index=fs.retrieve_index_3D(a,z_idx,m_idx)
    #print("For a=%i,b=%i,c=%i, our y index is %i"%(a,z_idx,m_idx,index))
    y_filtered.append(y[:,index])

y_filtered=np.array(y_filtered)
M=mass[m_idx]
z=z[z_idx]

#pick a fiducial theta value
A_idx_theta0=5
theta0=1.0

def derivative(profile_up,profile_low,delta):
    deriv=(profile_up-profile_low)/(2.*delta)
    return deriv

cmap=cm.get_cmap('viridis',len(delta_theta))
colors=cmap.colors
#just plot the data
fiducial_profile=y_filtered[A_idx_theta0,:]

fig=plt.figure(figsize=(6,8))
gs=gridspec.GridSpec(2,1,height_ratios=[2,1])
a0=plt.subplot(gs[0])
a1=plt.subplot(gs[1],sharex=a0)
plt.setp(a0.get_xticklabels(),visible=False)

for i in (A_idx_theta0-1,A_idx_theta0,A_idx_theta0+1):
    a0.semilogx(x,10**y_filtered[i,:]/10**fiducial_profile,label='%s = %.2f'%(vary_label,vary[i]))
for count,val in enumerate(delta_theta):
    params_up=[[theta0+val,z,M]]
    params_low=[[theta0-val,z,M]]

    profile_up=emulator(params_up).reshape(len(x))
    profile_low=emulator(params_low).reshape(len(x))

    deriv=derivative(profile_up,profile_low,val)
    a0.semilogx(x,10**profile_up/10**fiducial_profile,linestyle='dashed',color=colors[count],label=r'emu +$\Delta\theta=%.2f$'%val)
    a0.semilogx(x,10**profile_low/10**fiducial_profile,linestyle='dotted',color=colors[count])

    #plot the derivatives
    a1.semilogx(x,deriv,'-o',color=colors[count],label=r'$\Delta\theta=%.2f$'%val)
    a1.axhline(0,linestyle='dashed',color='gray',alpha=0.6)

#if we wanted individual titles
#plt.setp([a0,a1],title='test')
plt.suptitle(r'Vary %s, M = %.1f, z = %.2f, $\theta_0$ = 1.0'%(vary_label,M,z))
a1.set_xlabel('R (Mpc)',size=14)
a0.set_ylabel(ylabel,size=18)
a1.set_ylabel('Derivative',size=12)
a0.legend()
a1.legend()
gs.tight_layout(fig,rect=[0,0,1,0.97])
plt.savefig('./Derivative/'+suite+'_'+vary_str+'_M'+str(m_idx)+'_z'+str(z_idx)+'_'+prof+'.png',bbox_inches='tight')
plt.close()

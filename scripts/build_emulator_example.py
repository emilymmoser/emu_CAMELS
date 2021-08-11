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
import helper_functions as fs
import sys
import pickle


home='/home/cemoser/Repositories/emu_CAMELS/Emulator_profiles/'
suite='SIMBA'
vary_str='ASN1'
prof='rho_mean' #rho_mean,rho_med,pth_mean,pth_med
func_str='linear' #this is the Rbf interpolation function

mass=fs.mass
mass_str=fs.mass_str
snap=fs.snap
z=fs.choose_redshift(suite)
vary,sims=fs.choose_vary(vary_str)
samples=fs.cartesian_prod(vary,z,mass) 
nsamp=samples.shape[0]

samples,x,y,emulator=fs.build_emulator_3D(home,suite,vary_str,prof,func_str)

#pickle.dump(emulator,open('test_'+suite+'_'+vary_str+'_'+prof+'.p','wb'))
#np.savez('name.npz',emulator=emulator)


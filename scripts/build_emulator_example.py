import helper_functions as fs

home='./emulator_profiles/' #point to your profiles
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

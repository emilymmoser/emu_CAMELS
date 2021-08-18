from numpy.core.fromnumeric import var
import corner
import math
from matplotlib import pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy import integrate

from scripts import helper_functions as fs

import xray_emissivity

from colossus.cosmology import cosmology
from colossus.halo import mass_defs, concentration
COSMO = cosmology.setCosmology('planck18')

FIDUCIAL_FEEDBACK_PARAMETER = 1 # Fiducial value for AGN1/2, SN1/2

HUBBLE = 0.6711
Msun = 1.99e33
kpc = 3.0856e21
Mpc = 1e3*kpc
kb = 1.38e-16
mu = 0.59
X_H = 0.76
mp = 1.67e-24 # grams
me = 9.1 * 10**(-28) # grams

c = 3e8 # m / s

sigma_thomson = 6.65*10**(-29) # m^2

mue = 2/(1+X_H)

erg2keV = 6.242e8 #convert from erg to keV
cgs_density = Msun * kpc**(-3) # from Msun kpc^-3 to g cm^-3
# from Msol (km/s)**2 to keV cm^-3 
pressure_conversion_factor = 1 / kpc * erg2keV
Zsun = 0.0127
temperature_conversion_factor = (1e5)**2 * kb * erg2keV

rad2arcmin = 180.0/math.pi * 60.0
ster2sqdeg = 3282.80635
ster2sqarcmin = ster2sqdeg * 3600.0
ster2sqarcsec = ster2sqdeg * 3600.0 * 3600.0

#p_to_y = pe_factor * sigma_T / m_e_keV # TODO get values


# Cache emulators to save time
EMULATORS = {} # Key: (profile name, feedback parameter, simulation suite)


def build_emulator(profile_name, varied_feedback_parameter, simulation_suite):
    global EMULATORS

    emulator = EMULATORS.get((profile_name, varied_feedback_parameter, simulation_suite), None)
    if emulator is not None:
        return emulator

    home='./emulator_profiles/' #point to your profiles
    func_str='linear' #this is the Rbf interpolation function

    mass=fs.mass
    z=fs.choose_redshift(simulation_suite)
    vary,sims=fs.choose_vary(varied_feedback_parameter)
    samples=fs.cartesian_prod(vary,z,mass) 

    samples,x,y,emulator=fs.build_emulator_3D(home,simulation_suite,varied_feedback_parameter,profile_name,func_str)

    EMULATORS[(profile_name, varied_feedback_parameter, simulation_suite)] = emulator

    return emulator

def abel_projection(r2D, prof) :
    
    '''
    return: np array with projected profile
    '''
    
    prof2D = np.zeros(len(r2D))
    prof2D = np.zeros(prof.shape)
    prof3D = interp1d(r2D, prof, axis=1, kind='linear',fill_value='extrapolate')

    for irp, rp in enumerate(r2D) :
        integ = 0.0 
        zmax = np.sqrt(max(r2D)**2 + rp**2)
        zbin_edge = np.linspace(0.0, zmax, 10000)
        zbin = 0.5*(zbin_edge[1:] + zbin_edge[:-1])
        dz = zbin_edge[1:]-zbin_edge[:-1]
        rprime = np.sqrt(rp**2+zbin**2)

        fz = 2*prof3D(rprime)
        integ = integrate.trapezoid(fz,zbin)
        prof2D[:, irp] = integ

    return prof2D


xray = xray_emissivity.XrayEmissivity()
xray.read_emissivity_table('etable_05_2keV_cnts.hdf5')
eff_area = 2000.0 #cm^2

def compute_xray_profiles(halo_redshift, radii, density_profile, temperature_profile, metallicity_profile):
    np_xray_emissivity = xray.return_interpolated_emissivity(
        temperature_profile,
        metallicity_profile
    )

    #number densities of electron and H
    nH = density_profile * X_H / mp
    ne = density_profile / (mue * mp)

    np_xray_emissivity *= ne * nH * eff_area
             
    # Project X-ray emmissivity to get X-ray surface brightness
    np_xsb = abel_projection(radii * kpc , np_xray_emissivity ) / (4.0*math.pi)
    
    # account for redshift dimming, change units from per steradians to per arcmin^2
    np_xsb *= 1.0 /(1+halo_redshift)**4.0 /ster2sqarcmin 

    return (np_xray_emissivity, np_xsb)


RADII = np.array([
    1.633852131856735820e-02,
    2.839309588794570668e-02,
    4.934154556483250076e-02,
    8.574577877434103046e-02,
    1.490090854158103439e-01,
    2.589481121267818153e-01, 
    4.500002438570074315e-01,
    7.820107967121281423e-01,
    1.358979010617297423e+00,
    2.361634850903744809e+00,
    4.104051001103932172e+00,
    7.132023230948114190e+00,
    1.239403588140144663e+01,
]) * 1e3 # Convert from Mpc to kpc

def get_xsb_for_parameter(A_name, simulation_suite, A_value, z, M200c):
    # Build emulators
    print("Creating density emulator...")
    density_emulator = build_emulator("rho_med", A_name, simulation_suite)
    print("Creating temperature emulator...")
    temperature_emulator = build_emulator("temp_med", A_name, simulation_suite)
    print("Creating metallicity emulator...")
    metallicity_emulator = build_emulator("metal_med", A_name, simulation_suite)

    # Create profiles from emulator
    halo_emulate_param = [[A_value, z, M200c]]
    
    density_profile = np.power(10.0, density_emulator([halo_emulate_param])).flatten()
    temperature_profile = np.power(10.0, temperature_emulator([halo_emulate_param])).flatten()
    metallicity_profile = np.power(10.0, metallicity_emulator([halo_emulate_param])).flatten()

    # Convert profiles from cgs to expected units
    temperature_profile *= temperature_conversion_factor # Convert to keV
    metallicity_profile /= Zsun # Convert to Zsun


    emissivity_profile, xsb_profile = compute_xray_profiles(
        z,
        RADII,
        np.array([density_profile]),
        np.array([temperature_profile]),
        np.array([metallicity_profile]),
    ) 

    emissivity_profile = emissivity_profile[0]
    xsb_profile = xsb_profile[0]

    return xsb_profile


def get_y_for_parameter(A_name, simulation_suite, A_value, z, M200c):
    # Build emulator
    print("Creating pressure emulator...")
    pressure_emulator = build_emulator("PRESSURE TODO", A_name, simulation_suite)

    # Create profiles from emulator
    halo_emulate_param = [[A_value, z, M200c]]
    
    pressure_profile = np.power(10.0, pressure_emulator([halo_emulate_param])).flatten()

    # Convert profile from pressure to y
    y_profile = p_to_y * pressure_profile * Mpc # TODO get p_to_y
    y_profile = abel_projection(RADII * kpc, y_profile)

    return y_profile















##### Fisher Matrices and Corner Plots
def fisher_matrix_xsb(radii, params, delta=1e-2, instrument_resolution_arcmin=0.4, instrument_sensitivity=2e-3**2, log_halo_mass=13, halo_redshift=0.1, simulation_suite="SIMBA") :
    # Determine var based on instrument responses
    DA = COSMO.angularDiameterDistance(halo_redshift) / HUBBLE # Mpc
    cvir = concentration.concentration(10**log_halo_mass, 'vir', halo_redshift, model='diemer15')
    M500c, R500c, c500c = mass_defs.changeMassDefinition(10**log_halo_mass, cvir, halo_redshift, 'vir', '500c')
    M200m, R200m, c200m = mass_defs.changeMassDefinition(10**log_halo_mass, cvir, halo_redshift, 'vir', '200m')

    theta_500c = np.arctan((R500c)/DA) * 180.0 / math.pi * 60
    theta_200m = np.arctan((R200m)/DA) * 180.0 / math.pi * 60

    nbins = int(2*theta_200m / instrument_resolution_arcmin) + 1
    radial_arcmin_range = np.linspace(instrument_resolution_arcmin, theta_500c, nbins)
    rad_bins = (radial_arcmin_range/ 60.) * (math.pi / 180.0) * DA # Mpc
    rad_bins *= 1000 # Convert from Mpc to kpc

    var = np.full(rad_bins.shape,  instrument_sensitivity)

    # Initialize empty Fisher matrix and gradient matrix
    f = np.zeros([len(params), len(params)], dtype=np.float64)
    grad = np.zeros([len(params), len(rad_bins)], dtype=np.float64)

    for ind, param in enumerate(params):
        param_value = FIDUCIAL_FEEDBACK_PARAMETER
        param_less = (1.0 - delta) * param_value
        param_more = (1.0 + delta) * param_value

        h = 2.0*delta*param_value

        pro_less = get_xsb_for_parameter(param, simulation_suite, param_less, halo_redshift, log_halo_mass)
        pro_more = get_xsb_for_parameter(param, simulation_suite, param_more, halo_redshift, log_halo_mass)

        # Interpolate profiles to rad_bins
        pro_less_interp = interp1d(radii, pro_less, fill_value="extrapolate")(rad_bins)
        pro_more_interp = interp1d(radii, pro_more, fill_value="extrapolate")(rad_bins)

        diff = (pro_more_interp - pro_less_interp)/h

        grad[ind,:] = diff

    for ind1, _ in enumerate(params) :
        for ind2, _ in enumerate(params) :

            fij = 0.0

            for il, _ in enumerate(rad_bins) :
                fij += (grad[ind1,il]*grad[ind2,il]) / var[il]

                if fij == 0.0:
                    fij = 1.e-100

            f[ind1, ind2] = fij

    return f





# Which feedback parameters to vary
feedback_parameters = ["AAGN1", "AAGN2", "ASN1", "ASN2"]

log_halo_mass = 13
halo_redshift = 0.1
simulation_suite = "SIMBA"

f = fisher_matrix_xsb(
    RADII,
    feedback_parameters,
    delta=1e-2,
    instrument_sensitivity=1e-9,
    log_halo_mass=log_halo_mass,
    halo_redshift=halo_redshift,
    simulation_suite=simulation_suite
)

# Make corner plot
mean = np.ones(len(feedback_parameters))
covariance = np.linalg.inv(f)
print(covariance)

chain = np.random.multivariate_normal(mean, covariance, size=10000)

fig = corner.corner(
    chain,
    labels=feedback_parameters,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
)

fig.suptitle("$log_{10} (M_{200c} / M_{\odot}) = $" + f"{log_halo_mass}, z = {halo_redshift}, sim = {simulation_suite}")

#plt.savefig(f"./corner_plot_Mass{log_halo_mass}Redshift{halo_redshift}_{simulation_suite}.pdf")

plt.show()

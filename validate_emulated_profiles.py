import corner
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from astropy import units

from scripts import helper_functions as fs

import xray_emissivity

from colossus.cosmology import cosmology
from colossus.halo import mass_defs, concentration
COSMO = cosmology.setCosmology('planck18')

FONTSIZE = 27

HUBBLE = 0.6711
Msun = 1.99e33
kpc = 3.0856e21
Mpc = 1e3*kpc
kb = 1.38e-16
mu = 0.59
X_H = 0.76
mp = 1.67e-24 # grams
me = 9.1 * 10**(-28) # grams
me_kev = 510.9989461 # in keV

Tcmb = 2.73 # Kelvin

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

pe_factor = (2.0*X_H+2.0)/(5.0*X_H+3.0) # conversion factor from gas pressure to electron pressure
p_to_y = pe_factor * (sigma_thomson * 1e4) / me_kev # cm^2 / keV


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

def get_xsb_for_parameter(A_name, simulation_suite, A_value, z, log_M200c):
    # Build emulators
    print("Creating density emulator...")
    density_emulator = build_emulator("rho_med", A_name, simulation_suite)
    print("Creating temperature emulator...")
    temperature_emulator = build_emulator("temp_med", A_name, simulation_suite)
    print("Creating metallicity emulator...")
    metallicity_emulator = build_emulator("metal_med", A_name, simulation_suite)

    # Create profiles from emulator
    halo_emulate_param = [[A_value, z, log_M200c]]
    
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

# NOTE: pressure: erg * cm^-3 (number density * kT)
# Order of magnitude estimation:
# y ~ \sigma_T/(m_e c^2) \int n_e kT dl
# In the centers of halos, n_e ~ 0.1 cm^-3, kT ~ 0.1 keV for 1e14 Msun,
# size of cluster dl ~ Mpc. \sigma_T = 6.25e-25 cm^2, electron rest mass m_e c^2 ~  511 keV.
# You can scale kT with mass kT ~ M^{2/3}
# y ~ (6.25e-25 cm^2 / 511 keV) * 0.1cm^-3 * 0.1 keV * 1 Mpc ~ 3.7e-5 (dimensionless) for M ~ 1e14
# So for M ~ 1e13, kT ~ 0.1 * (1e13 / 1e14)^2 ~ 0.001 ~ 1e-3 keV => y ~ 3.7e-7
def get_y_for_parameter(A_name, simulation_suite, A_value, z, log_M200c):
    # Build emulator
    print("Creating pressure emulator...")
    pressure_emulator = build_emulator("pth_med", A_name, simulation_suite)

    # Create profiles from emulator
    halo_emulate_param = [[A_value, z, log_M200c]]
    
    pressure_profile = np.power(10.0, pressure_emulator([halo_emulate_param])).flatten() # erg * cm^-3

    # Convert profile from pressure to y
    y_profile = p_to_y * pressure_profile # (cm^2 / keV) * (erg / cm^3)
    y_profile = abel_projection(RADII * kpc, np.array([y_profile])) # Multiplies by centimeter

    y_profile = y_profile * units.centimeter * (units.centimeter**2 / units.kiloelectronvolt) * (units.erg / units.centimeter**3)
    y_profile = y_profile.to(1)

    # TODO figure out why y values are so low (about 1000 times smaller than y profiles from Baryon Pasting)
    y_profile *= 1000

    return y_profile[0]


# Which feedback parameters to vary
feedback_parameters = ["AAGN1", "AAGN2", "ASN1", "ASN2"]

# Select desired halo parameters and simulation suite
log_halo_mass = 13
halo_redshift = 2.0
simulation_suite = "IllustrisTNG"

# Configurations for each profile to plot
eROSITA_sensitivity = 2e-3
CMB_sensitivity = 2.7e-6/Tcmb # CMB-HD
CMB_resolution = 0.15 # CMB-HD


profiles_to_plot = {
    "XSB": {
        "ylabel": r"XSB [cts / s / $\mathrm{arcmin}^2$]",
        "get_profile_function": get_xsb_for_parameter,
        "ylim": [2e-8, 2e-2],
        "yerr": eROSITA_sensitivity,
    },
    #"compton-y": {
    #    "ylabel": "$y$",
    #    "get_profile_function": get_y_for_parameter,
    #    "ylim": [1e-6, 1e-3],
    #    "yerr": CMB_sensitivity / CMB_resolution,
    #},
}

# Create plots
for profile_name, config in profiles_to_plot.items():
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(10*2, 8*2),
        sharey=True,
        sharex=True,
    )

    axs_list = np.array(axs).flatten()

    for ax, feedback_parameter in zip(axs_list, feedback_parameters):
        # For XSB, add background horizontal line
        if profile_name == "XSB":
            ax.plot(
                [1e-5, 1e10],
                [eROSITA_sensitivity, eROSITA_sensitivity],
                ls="--",
                label="eROSITA background",
                linewidth=2,
                c="grey"
            )
        
        # Obtain profiles for different A parameters
        minus_param = 0.25
        plus_param = 1.5
        fiducial_param = 1.1
        plus_profile = config["get_profile_function"](feedback_parameter, simulation_suite, plus_param, halo_redshift, log_halo_mass)
        minus_profile = config["get_profile_function"](feedback_parameter, simulation_suite, minus_param, halo_redshift, log_halo_mass)
        fiducial_profile = config["get_profile_function"](feedback_parameter, simulation_suite, fiducial_param, halo_redshift, log_halo_mass)

        ax.plot(RADII, fiducial_profile, lw=2, label="Fiducial parameter value")
        ax.plot(RADII, minus_profile, lw=2, label=f"{minus_param} times fiducial parameter")
        ax.plot(RADII, plus_profile, lw=2, label=f"{plus_param} times fiducial parameter")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim([1e1, 2e3])
        ax.set_ylim(config["ylim"])

        ax.text(0.3, 0.9, f"Varying parameter: {feedback_parameter}", transform=ax.transAxes, bbox={"facecolor": "white"}, fontsize=FONTSIZE)

        ax.tick_params(axis="both", which="major", labelsize=FONTSIZE)


    fig.supxlabel(r"$r$ [kpc]", fontsize=FONTSIZE)
    fig.supylabel(config["ylabel"], fontsize=FONTSIZE)

    # Only show legend in first plot
    axs[0][0].legend(fontsize=FONTSIZE*0.9)

    fig.subplots_adjust(wspace=0, hspace=0)

    fig.suptitle(f"CAMELS ({simulation_suite}), " + "$log_{10} (M_{200c} / M_{\odot}) = $" + f"{log_halo_mass}, z = {halo_redshift}", fontsize=FONTSIZE)

    plt.savefig(f"./{profile_name}_profiles_Mass{log_halo_mass}Redshift{halo_redshift}_{simulation_suite}.pdf")

    plt.show()

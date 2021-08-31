# emu_CAMELS

emu_CAMELS is a repository creating an emulator for the CAMELS density and pressure profiles. 

A different emulator is created for each simulation suite (SIMBA/IllustrisTNG), feedback parameter (ASN1/2, AAGN1/2) and profile (rho mean/median, pth mean/median).

The emulator is constructed using the profiles found in emulator_profiles (but this can be substituted for the CAMELS SQL database, for example) and the interpolation methods of the repository *ostrich*, found at: https://github.com/dylancromer/ostrich.

## Setup
```
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

**NOTE** that `ostrich` is already included in this repo, due to a change in `ostrich/ostrich/interpolate.py` that allows for the emulator to run. When the main `ostrich` repo gets fixed, we can remove the `ostrich` directory from this repo, and add the corresponding line to `requirements.txt`.



## A few additional notes regarding the Emulator
1. The input masses are in log, so if you want to emulate a profile of mass 1e14 Msol, you would input it as 14.

2. The current profiles cover the radial range 0.01-10 Mpc. This can also be seen by looking at the txt files in the emulator_profiles directory.

3. The emulator will return a profile in log values, and CGS units. 

4. An emulator is a fancy interpolator, so we can only ask it to return profiles within the ranges of each parameter (feedback value, redshift, and mass). The current ranges for these parameters can be found in scripts/helper_functions.

## Luis - Fisher Forecast Analysis and Emulator Profile Validation

After executing the setup above, you can:

- Visualize XSB and Compton-y profiles generated from the emulated profiles. Run `python validate_emulated_profiles.py`.
- Generate Fisher forecasts for both eROSITA and CMB surveys, using the emulated XSB and Compton-y profiles. Run `python fisher_forecasts.py`.

**NOTE** that you can modify the halo mass, halo redshift, and the profiles or surveys used in the corner plot, by editing the appropriate Python files above.
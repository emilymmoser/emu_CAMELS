# emu_CAMELS

emu_CAMELS is a repository creating an emulator for the 1P and LH sets CAMELS density and pressure profiles. 

For the 1P set, a different emulator is created for each simulation suite (SIMBA/IllustrisTNG), feedback parameter (ASN1/2, AAGN1/2) and profile (rho mean/median, pth mean/median).

The 1P emulators are constructed using the profiles found in emulator_profiles and the interpolation methods of the repository *ostrich*, found at: https://github.com/dylancromer/ostrich.

Since the LH set is much larger we do not provide a copy of all the profiles, but we include a script to extract the profile information from the hdf5 files (provided in the CAMELS data release: https://camels.readthedocs.io/en/latest/data_access.html) in scripts/LH_emulator/extract_from_hdf5.py. 

## Dependencies:

numpy, scipy

ostrich - should download its own dependencies when cloning the repo, see note 5 below for temporary fix

## Example:

Run the example script for the 1P emulator, scripts/1P_emulator/build_emulator_example.py
Specify the suite, feedback parameter, and profile type in lines 3-6.

Run the example script for the LH emulator, scripts/LH_emulator/build_emulator_LH.py. 

## A few additional notes
1. The input masses are in log, so if you want to emulate a profile of mass 1e14 Msol, you would input it as 14.

2. The current profiles cover the radial range ~(3e-3 for TNG,5e-4 for SIMBA)-12 Mpc. This can also be seen by looking at the txt files in the emulator_profiles directory.

3. The emulator will return a profile in log values, and CGS units. 

4. An emulator is a fancy interpolator, so we can only ask it to return profiles within the ranges of each parameter (feedback value, redshift, and mass). The current ranges for these parameters can be found in scripts/helper_functions.

5. I changed the Rbf interpolater class in ostrich/interpolate.py to have an additional input (the interpolation function, it was set to multiquadric before) in my fork of *ostrich* which is currently a pull request, make sure you update these lines before using this repository! (update when merged)
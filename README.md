# emu_CAMELS

emu_CAMELS is a repository creating emulators for density and pressure profiles of the 1P and LH sets within the CAMELS simulation suite. 

For the 1P set, a different emulator is created for each simulation suite (SIMBA/IllustrisTNG), feedback parameter (ASN1/2, AAGN1/2) and profile (rho mean/median, pth mean/median). The LH set varies all of these parameters at once.

The general 1P and LH emulators are constructed using the profiles found in emulator_profiles. These profiles are interpolated using the repository *ostrich*, found at: https://github.com/dylancromer/ostrich.

We include an example script to extract the profile information from the hdf5 files (provided in the CAMELS data release: https://camels.readthedocs.io/en/latest/data_access.html) in scripts/LH_emulator/extract_from_hdf5.py.



## Dependencies:

numpy, scipy

ostrich - should download its own dependencies when cloning the repository

## Examples and Tests:
The redshifts (and corresponding snapshots) and mass ranges are defined at the beginning of scripts/helper_functions* for both emulator types. Adjust accordingly for the values you want. In general, there are two different types of emulators: general and weighted (CMASS). The general emulators take in an average unweighted profile for each mass bin, seen by the "uw" filenames in the load_profiles functions. The weighted emulator (CMASS) takes in an average profile for the entire mass range weighted by the CMASS mass distribution, seen by the "w" filenames. 

Run the example script for the general 1P emulator, scripts/1P_emulator/build_emulator_example.py
Specify the suite, feedback parameter, and profile type in lines 3-6.

Run the example script for the general LH emulator, scripts/LH_emulator/build_emulator_LH.py. NOTE: I have only uploaded the profiles for a single snapshot (024) because there are a lot. To add other redshifts, you will need to run scripts/LH_emulator/extract_from_hdf5.py, and uncomment line 13 in scripts/LH_emulator/helper_functions_LH.py to use all of the snapshots rather than only '024'.  

And for the CMASS-specific LH emulator, scripts/LH_emulator/build_emulator_LH_CMASS.py. This is built off profiles weighted by the CMASS mass distribution, only for snapshot 024.  

The accuracy of the emulators can be plotted in the tests/radial_errors_general_emu.py (for 1P) and tests/drop1_test_LH_emulator.py (for LH) scripts.

## A few additional notes
1. The input masses are in log(solar masses), so if you want to emulate a profile of mass 1e14 Msol, you would input it as 14.

2. The emulator will return a profile in log values and CGS units.

3. The current 1P profiles cover the radial range ~(3e-3 for TNG,5e-4 for SIMBA)-12 Mpc. This can also be seen by looking at the txt files in the emulator_profiles directory. The LH profiles were not computed as far out, stopping at ~1-2 Mpc.

4. An emulator is a fancy interpolator, so we can only ask it to return profiles within the ranges of each parameter (feedback value, redshift, and mass).
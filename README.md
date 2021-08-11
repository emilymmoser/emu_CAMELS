# emu_CAMELS

emu_CAMELS is a repository creating an emulator for the CAMELS density and pressure profiles. 

A different emulator is created for each simulation suite (SIMBA/IllustrisTNG), feedback parameter (ASN1/2, AAGN1/2) and profile (rho mean/median, pth mean/median).

The emulator is constructed using the profiles found in emulator_profiles (but this can be substituted for the CAMELS SQL database, for example) and the interpolation methods of the repository *ostrich*, found at: https://github.com/dylancromer/ostrich.


## Dependencies:

numpy, scipy

ostrich - should download its own dependencies when cloning the repo

## Example:

Run the example script, scripts/build_emulator_example.py

Specify the suite, feedback parameter, and profile type in lines 3-6.
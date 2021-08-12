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

## A few additional notes
1. The input masses are in log, so if you want to emulate a profile of mass 1e14$M_\odot$, you would input it as 14.

2. The current profiles cover the radial range $0.01 \leq R (Mpc) \leq 10$. This can also be seen by looking at the txt files in the emulator_profiles directory.

3. The emulator will return a profile in log values, and CGS units. 

4. An emulator is a fancy interpolator, so we can only ask it to return profiles within the ranges of each parameter (feedback value, redshift, and mass). The current ranges for these parameters can be found in scripts/helper_functions.

### Note to Luis:

I added an interpolation class to *ostrich* that you can see being used in helper_functions.py, in the build_emulator_3d function. If you run this example as is you'll get an error message saying something like "ostrich.interpolate doesn't have class RbfInterpolator". If we go this route of you running the code yourself, I can send you the interpolation class to add to the file in *ostrich*. 

Eventually I'll need to make a pull request for this, but the grad student who wrote it is on leave right now so we're kinda stuck. I guess I could also make my own fork and link it here.. I'll think more about that later. It's not a huge deal for now, just a few additional lines. 
.. _latest-news:

Latest News
===========

I'm going to attempt to keep this page updated with news about the code, including updates to the code and documentation, as well as issues people make me aware of.

November 2025
------------

**Tuesday 18th:** Version 1.3.2 of the code has been released today.

This includes a variety of new updates and bug fixes, most notably:

 - The nebular models have been re-run with Cloudy v25.00 and the "grains ISM" command has been removed from the Cloudy config files used to make the grid. This is because, at high ionization parameters (e.g., logU>-2), a lot of energy was being diverted from emission lines to IR thermal radiation from warm dust grains. This was leading to significantly fainter lines in spectra with high ionization parameter (e.g., up to an order of magnitude fainter than predicted by the Kennicutt 2012 relationship). Line ratios are largely unaffected, but lines are now up to ~1 dex brighter at high ionization parameters.

 - A bug in the way the Draine + Li (2007) dust emission models were implemented has been fixed. The gamma parameter was previously being incorrectly applied in a light-weighted sense rather than mass-weighted.

 - The code now includes a variety of new modelling options from `Leung et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.4029L/abstract>`_, including a new variety of Gaussian process noise model and various options for time-evolution of stellar metallicity.

 - Improvements and bug fixes to the damped Lyman alpha system model.

 - Escape fraction can now be varied with fit_info["nebular"]["fesc"], still defaults to zero.

 - The `Wild et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007MNRAS.381..543W/abstract>`_ dust attenuation model is now available.

 - A variety of improved error/warning messages for common mistakes and issues.

April 2023
------------

**Tuesday 11th:** Version 1.0.1 of the code has been released today.

The code now includes the ability for the user to specify a custom spectral resolving power curve as a function of wavelength, which will be convolved with spectral models prior to fitting to data.

This has been included with the help of several people in response largely to the flood of data being produced by the JWST NIRSpec prism. You can find a tutorial on how to implement this in the examples folder on GitHub under Further Examples 3.

December 2022
------------

**Thursday 8th:** Version 1.0.0 of the code has been released today.

The code now includes the ability to fit the Leja et al. (2019) continuity non-parametric star-formation history model. You can find a tutorial on this in the examples folder on GitHub under Further Examples 2.


June 2021
------------

**Wednesday 16th:** Version 0.8.6 of the code has been released today.

This update fixes a bug when specifying low velocity dispersions, which caused spectroscopy to be multiplied by a factor when the veldisp argument was supplied (e.g. 1.002 at 150 km/s, 1.09 at 100 km/s). This had no effect on photometry, or any models that didn't have the "veldisp" argument set. Thanks to Phil Short, Hin Leung and Vivienne Wild for pointing this out.

It's also recently been brought to my attention by Renske Smit that unexpected things can happen when asking the code to produce spectra with very low sampling (e.g. something like spec_wavs = np.arange(5000., 1000., 100.)). See Renske's issue on the GitHub repository for more info on this.

I've also done some more updating of the spectral index fitting functionality, which isn't documented at the moment, but I've now managed to get working for a few different problems.


January 2021
------------

**Tuesday 12th:** Version 0.8.5 of the code has been released today.

This fixes a few bugs with lesser-used parts of the code, in particular a fix to the likelihood function for spectral index fitting and a fix for the line_fluxes dictionary (with thanks to Christian Binggeli). I've also added more customisation options to a few of the plotting functions.

A new version of pymultinest has now been released that includes the options required for bagpipes to do serial fitting within an MPI environment, so there's no longer any need to download my modified version as long as you're using bagpipes >= v0.8.5 with pymultinest >= v2.11.

The new release of pandas (v1.2.0) has broken compatibility with deepdish (with thanks to Ariel Werle for pointing this out), so pandas <= v1.1.5 has been added as a requirement for bagpipes until this gets fixed.

I'm going to be working on a review of the current examples and documentation, as well as attempting to put together some "advanced topics" Jupyter notebooks for the repository. Let me know if there are specific things you'd like to see covered in these and I'll do my best to include them.

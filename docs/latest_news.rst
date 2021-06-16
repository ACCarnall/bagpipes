.. _latest-news:

Latest News
===========

I'm going to attempt to keep this page updated with news about the code, including updates to the code and documentation, as well as issues people make me aware of.


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

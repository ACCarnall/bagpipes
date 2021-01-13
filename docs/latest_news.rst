.. _latest-news:

Latest News
===========

I'm going to attempt to keep this page updated with news about the code, including updates to the code and documentation, as well as issues people make me aware of.

January 2021
------------

**Tuesday 12th:** Version 0.8.5 of the code has been released today.

This fixes a few bugs with lesser-used parts of the code, in particular a fix to the likelihood function for spectral index fitting and a fix for the line_fluxes dictionary (with thanks to Christian Binggeli). I've also added more customisation options to a few of the plotting functions.

A new version of pymultinest has now been released that includes the options required for bagpipes to do serial fitting within an MPI environment, so there's no longer any need to download my modified version as long as you're using bagpipes >= v0.8.5 with pymultinest >= v2.11.

The new release of pandas (v1.2.0) has broken compatibility with deepdish (with thanks to Ariel Werle for pointing this out), so pandas <= v1.1.5 has been added as a requirement for bagpipes until this gets fixed.

I'm going to be working on a review of the current examples and documentation, as well as attempting to put together some "advanced topics" Jupyter notebooks for the repository. Let me know if there are specific things you'd like to see covered in these and I'll do my best to include them.

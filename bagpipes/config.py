from __future__ import print_function,  division,  absolute_import

import os
import numpy as np

from astropy.io import fits

from .utils import *
from .models.making import igm_inoue2014

""" This file contains all of the configuration variables for Bagpipes.
This includes loading different grids of models into the code, and the
way the spectral sampling for models is constructed. The default values
are a trade-off between speed and accuracy, you may find changing them
negatively affects one or both of these things. """


""" These variables control the wavelength sampling for models. """

# Sets the maximum redshift the code is set up to calculate models for.
max_redshift = 10.

# Sets the R = lambda/dlambda value for spectroscopic outputs.
R_spec = 600.

# Sets the R = lambda/dlambda value for photometric outputs.
R_phot = 100.

# Sets the R = lambda/dlambda value for other spectral regions.
R_other = 20.


""" These variables control the age sampling for the stellar and nebular
emission models. The stellar models will be automatically rebinned onto
the chosen age grid. The nebular models must be re-computed with Cloudy
if these variables are changed. """

# Sets the default age sampling for stellar models in log10(Gyr).
# Beware: if you change this you need to recompute the nebular models.
age_sampling = np.arange(6., np.log10(cosmo.age(0.).value) + 9., 0.1)

# Set up edge positions for age bins for stellar + nebular models.
age_bins = 10**make_bins(age_sampling, make_rhs=True)[0]
age_bins[0] = 0.
age_bins[-1] = 10**9*cosmo.age(0.).value

# Set up widths for the age bins for the stellar + nebular models.
age_widths = age_bins[1:] - age_bins[:-1]

# Convert the age sampling from log10(Gyr) to Gyr.
age_sampling = 10**age_sampling


""" These variables tell the code where to find the raw stellar emission
models, as well as some of their basic properties. """

try:
    # Name of the fits file storing the stellar models
    stellar_file = "bc03_miles_stellar_grids.fits"

    # The metallicities of the stellar grids in units of Z_Solar
    metallicities = np.array([0.005, 0.02, 0.2, 0.4, 1., 2.5, 5.])

    # The wavelengths of the grid points in Angstroms
    wavelengths = fits.open(grid_dir + "/" + stellar_file)[-1].data

    # The ages of the grid points in Gyr
    raw_stellar_ages = fits.open(grid_dir + "/" + stellar_file)[-2].data

    # The fraction of stellar mass still living (1 - return fraction).
    # Axis 0 runs over metallicity, axis 1 runs over age.
    live_frac = fits.open(grid_dir + "/" + stellar_file)[-3].data[:, 1:]

    # The raw stellar grids, stored as a FITS HDUList.
    # The different HDUs are the grids at different metallicities.
    # Axis 0 of each grid runs over wavelength, axis 1 over age.
    raw_stellar_grid = fits.open(grid_dir + "/" + stellar_file)[1:8]

    # Set up edge positions for metallicity bins for stellar models.
    metallicity_bins = make_bins(metallicities, make_rhs=True)[0]
    metallicity_bins[0] = 0.
    metallicity_bins[-1] = 10.

except IOError:
    print("Failed to load stellar grids, these should be placed in"
          + " the bagpipes/models/grids/ directory.")


""" These variables tell the code where to find the raw nebular emission
models, as well as some of their basic properties. """

try:
    # Names of files containing the nebular grids.
    neb_cont_file = "bc03_miles_nebular_cont_grids.fits"
    neb_line_file = "bc03_miles_nebular_line_grids.fits"

    # Names for the emission features to be tracked.
    line_names = np.loadtxt(grid_dir + "/cloudy_lines.txt",
                            dtype="str", delimiter="}")

    # Wavelengths of these emission features in Angstroms.
    line_wavs = np.loadtxt(grid_dir + "/cloudy_linewavs.txt")

    # Ages for the nebular emission grids.
    neb_ages = fits.open(grid_dir
                         + "/" + neb_line_file)[1].data[1:, 0]

    # Wavelengths for the nebular continuum grids.
    neb_wavs = fits.open(grid_dir + "/" + neb_cont_file)[1].data[0, 1:]

    # LogU values for the nebular emission grids.
    logU = np.arange(-4., -1.99, 0.5)

    # Grid of line fluxes.
    line_grid = fits.open(grid_dir + "/" + neb_line_file)

    # Grid of nebular continuum fluxes.
    cont_grid = fits.open(grid_dir + "/" + neb_cont_file)

except IOError:
    print("Failed to load nebular grids, these should be placed in the"
          + " bagpipes/models/grids/ directory.")


""" These variables tell the code where to find the raw dust emission
models, as well as some of their basic properties. """

try:
    # Values of Umin for each of the Draine + Li (2007) dust emission grids.
    umin_vals = np.array([0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 1.00,
                          1.20, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00, 8.00,
                          10.0, 12.0, 15.0, 20.0, 25.0])

    # Values of qpah for each of the Draine + Li (2007) dust emission grids.
    qpah_vals = np.array([0.10, 0.47, 0.75, 1.12, 1.49, 1.77,
                          2.37, 2.50, 3.19, 3.90, 4.58])

    # Draine + Li (2007) dust emission grids, stored as a FITS HDUList.
    dust_grid_umin_only = fits.open(grid_dir + "/dl07_grids_umin_only.fits")

    dust_grid_umin_umax = fits.open(grid_dir + "/dl07_grids_umin_umax.fits")

except IOError:
    print("Failed to load dust emission grids, these should be placed in the"
          + " bagpipes/models/grids/ directory.")

""" These variables tell the code where to find the raw IGM attenuation
models, as well as some of their basic properties. """

# Redshift points for the IGM grid.
igm_redshifts = np.arange(0.0, max_redshift + 0.01, 0.01)

# Wavelength points for the IGM grid.
igm_wavelengths = np.arange(1.0, 1225.01, 1.0)

# If the IGM grid has not yet been calculated, calculate it now.
if not os.path.exists(grid_dir + "/d_igm_grid_inoue14.fits"):
    igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

else:
    # Check that the wavelengths and redshifts in the igm file are right
    igm_file = fits.open(grid_dir + "/d_igm_grid_inoue14.fits")

    if len(igm_file) != 4:
        igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

    else:
        wav_check = np.min(igm_file[2].data == igm_wavelengths)
        z_check = np.min(igm_file[3].data == igm_redshifts)

        if not wav_check or not z_check:
            igm_inoue2014.make_table(igm_redshifts, igm_wavelengths)

# 2D numpy array containing the IGM attenuation grid.
raw_igm_grid = fits.open(grid_dir + "/d_igm_grid_inoue14.fits")[1].data


""" These variables are alternatives to those given in the stellar
section, they are for using the BPASS stellar population models.

    # Name of the fits file storing the stellar models
    stellar_file = "bpass_bin-imf135_300_stellar_grids.fits"

    # The metallicities of the stellar grids in units of Z_Solar
    metallicities = np.array([10**-5, 10**-4, 0.001, 0.002, 0.003, 0.004,
                              0.006, 0.008, 0.010, 0.014, 0.020, 0.030,
                              0.040])/0.02

    # The wavelengths of the grid points in Angstroms
    wavelengths = fits.open(grid_dir + "/" + stellar_file)[-1].data

    # The ages of the grid points in Gyr
    raw_stellar_ages = fits.open(grid_dir + "/" + stellar_file)[-2].data

    # The fraction of stellar mass still living (1 - return fraction).
    # Axis 0 runs over metallicity, axis 1 runs over age.
    live_frac = fits.open(grid_dir + "/" + stellar_file)[-3].data

    # The raw stellar grids, stored as a FITS HDUList.
    # The different HDUs are the grids at different metallicities.
    # Axis 0 of each grid runs over wavelength, axis 1 over age.
    raw_stellar_grid = fits.open(grid_dir + "/" + stellar_file)[1:14]
"""

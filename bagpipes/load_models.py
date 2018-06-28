from __future__ import print_function, division, absolute_import

import numpy as np
from astropy.io import fits
import os

install_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

bc03_miles_path = install_dir + "/pipes_models/stellar/bc03_miles/"
bpass_bin_path = install_dir + "/pipes_models/stellar/bpass_bin/"


def bc03_miles():
    """ Function for loading up the BC03 MILES model library from
    compressed fits versions. """
    zmet_vals = np.array([0.005, 0.02, 0.2, 0.4, 1., 2.5, 5.])
    modelwavs = fits.open(bc03_miles_path
                          + "bc03_miles_stellar_grids.fits")[-1].data

    # ages[0] = 0.
    ages = fits.open(bc03_miles_path
                     + "bc03_miles_stellar_grids.fits")[-2].data

    live_frac = fits.open(bc03_miles_path
                          + "bc03_miles_stellar_grids.fits")[-3].data

    return zmet_vals, modelwavs, ages, live_frac


def bc03_miles_get_grid(zmet_ind):
    return fits.open(bc03_miles_path
                     + "bc03_miles_stellar_grids.fits")[zmet_ind+1].data


def bpass_bin():
    """ Function for loading up the bpass binary model library from
    compressed fits versions. """
    zmet_vals = np.array([0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.006,
                          0.008, 0.010, 0.014, 0.020, 0.030, 0.040])/0.02

    modelwavs = fits.open(bpass_bin_path
                          + "bpass_bin_stellar_grids.fits")[-1].data

    ages = fits.open(bpass_bin_path
                     + "bpass_bin_stellar_grids.fits")[-2].data

    live_frac = fits.open(bpass_bin_path
                          + "bpass_bin_stellar_grids.fits")[-3].data

    return zmet_vals, modelwavs, ages, live_frac


def bpass_bin_get_grid(zmet_ind):
    return fits.open(bpass_bin_path
                     + "bpass_bin_stellar_grids.fits")[zmet_ind+1].data

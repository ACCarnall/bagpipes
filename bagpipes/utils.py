from __future__ import print_function, division, absolute_import

import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM


def make_dirs(run="."):
    """ Make local Bagpipes directory structure in working dir. """

    if not os.path.exists(working_dir + "/pipes"):
        os.mkdir(working_dir + "/pipes")

    if not os.path.exists(working_dir + "/pipes/plots"):
        os.mkdir(working_dir + "/pipes/plots")

    if not os.path.exists(working_dir + "/pipes/posterior"):
        os.mkdir(working_dir + "/pipes/posterior")

    if not os.path.exists(working_dir + "/pipes/cats"):
        os.mkdir(working_dir + "/pipes/cats")

    if run is not ".":
        if not os.path.exists("pipes/posterior/" + run):
            os.mkdir("pipes/posterior/" + run)

        if not os.path.exists("pipes/plots/" + run):
            os.mkdir("pipes/plots/" + run)


def make_bins(midpoints, make_rhs=False):
    """ A general function for turning an array of bin midpoints into an
    array of bin left hand side positions and bin widths. Splits the
    distance between bin midpoints equally in linear space.

    Parameters
    ----------

    midpoints : numpy.ndarray
        Array of bin midpoint positions

    make_rhs : bool
        Whether to add the position of the right hand side of the final
        bin to bin_lhs, defaults to false.
    """

    bin_widths = np.zeros_like(midpoints)

    if make_rhs:
        bin_lhs = np.zeros(midpoints.shape[0]+1)
        bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
        bin_widths[-1] = (midpoints[-1] - midpoints[-2])
        bin_lhs[-1] = midpoints[-1] + (midpoints[-1]-midpoints[-2])/2
        bin_lhs[1:-1] = (midpoints[1:] + midpoints[:-1])/2
        bin_widths[:-1] = bin_lhs[1:-1]-bin_lhs[:-2]

    else:
        bin_lhs = np.zeros_like(midpoints)
        bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
        bin_widths[-1] = (midpoints[-1] - midpoints[-2])
        bin_lhs[1:] = (midpoints[1:] + midpoints[:-1])/2
        bin_widths[:-1] = bin_lhs[1:]-bin_lhs[:-1]

    return bin_lhs, bin_widths


# Set up necessary variables for cosmological calculations.
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
z_array = np.arange(0., 100., 0.01)
age_at_z = cosmo.age(z_array).value
ldist_at_z = cosmo.luminosity_distance(z_array).value

install_dir = os.path.dirname(os.path.realpath(__file__))
grid_dir = install_dir + "/models/grids"
working_dir = os.getcwd()

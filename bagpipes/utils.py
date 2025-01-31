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

    if run != ".":
        if not os.path.exists("pipes/posterior/" + run):
            os.mkdir("pipes/posterior/" + run)
            #os.mkdirs("pipes/posterior/" + run, exist_ok = True) # updated by austind 14/10/23

        if not os.path.exists("pipes/plots/" + run):
            os.mkdir("pipes/plots/" + run)
            #os.mkdirs("pipes/plots/" + run, exist_ok = True) # updated by austind 14/10/23


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

# A dictionary to convert between inputted line names and the cloudy output keys
lines_dict = {
    "Halpha": "H  1  6562.81A",
    "Hbeta": "H  1  4861.33A",
    "Hgamma": "H  1  4340.46A",
    "OIII_5007": "O  3  5006.84A",
    "OIII_4959": "O  3  4958.91A",
    # "OII_3727": "O  2  3727.09A",
    # "OII_3729": "O  2  3729.88A",
    "NII_6548": "N  2  6548.05A",
    "NII_6584": "N  2  6583.45A",
    # "SII_6717": "S  2  6716.44A",
    # "SII_6731": "S  2  6730.82A",
    # "NeIII_3869": "Ne 3  3868.76A",
    # "NeIII_3968": "Ne 3  3967.47A",
    # "NeIII_3974": "Ne 3  3974.98A",
    # "NeIII_3342": "Ne 3  3342.18A",
    # "HeII_1640": "He 2  1640.42A",
    # "HeII_4686": "He 2  4685.68A",
    # "HeI_6678": "He 1  6678.15A",
    # "HeI_5876": "He 1  5875.62A",
    # "HeI_4471": "He 1  4471.48A",
}
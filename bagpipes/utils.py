from __future__ import print_function, division, absolute_import

import numpy as np
import os
import sys

from astropy.io import fits

from . import load_models

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
z_array = np.arange(0., 10., 0.01)
age_at_z = cosmo.age(z_array).value
ldist_at_z = cosmo.luminosity_distance(z_array).value

install_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
working_dir = os.getcwd()

allstellargrids = {}
allcloudylinegrids = {}
allcloudycontgrids = {}

if not os.path.exists(install_dir
                      + "/models/igm/d_igm_grid_inoue14.fits"):

    from . import igm_inoue2014
    igm_inoue2014.make_table()

igm_grid = fits.open(install_dir
                     + "/models/igm/d_igm_grid_inoue14.fits")[1].data

igm_redshifts = np.arange(0.0, 10.01, 0.01)
igm_wavs = np.arange(1.0, 1225.01, 1.0)

# line_labels: array of emission line labels
# line_wavs: array of emission line wavelengths
label_path = install_dir + "/models/nebular/cloudy_lines.txt"
wavs_path = install_dir + "/models/nebular/cloudy_linewavs.txt"
line_names = np.loadtxt(label_path, dtype="str", delimiter="\t")
line_wavs = np.loadtxt(wavs_path)

# Generate dicts containing information about spectral models and
# populate them using the set_model_type function.
gridwavs = {}
ages = {}
age_lhs = {}
age_widths = {}
zmet_vals = {}
zmet_lims = {}
live_frac = {}

model_type = "bc03_miles"
full_age_sampling = False
max_redshift = 10.

logU_grid = np.arange(-4., -1.99, 0.5)


def make_bins(midpoints, make_rhs=False):
    """ A function to take an array of bin midpoints and return an
    array of bin left hand side positions and widths. """
    bin_widths = np.zeros(midpoints.shape[0])

    if make_rhs:
        bin_lhs = np.zeros(midpoints.shape[0]+1)
        bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
        bin_widths[-1] = (midpoints[-1] - midpoints[-2])
        bin_lhs[-1] = midpoints[-1] + (midpoints[-1]-midpoints[-2])/2
        bin_lhs[1:-1] = (midpoints[1:] + midpoints[:-1])/2
        bin_widths[:-1] = bin_lhs[1:-1]-bin_lhs[:-2]

    else:
        bin_lhs = np.zeros(midpoints.shape[0])
        bin_lhs[0] = midpoints[0] - (midpoints[1]-midpoints[0])/2
        bin_widths[-1] = (midpoints[-1] - midpoints[-2])
        bin_lhs[1:] = (midpoints[1:] + midpoints[:-1])/2
        bin_widths[:-1] = bin_lhs[1:]-bin_lhs[:-1]

    return bin_lhs, bin_widths


# Controls the grid of ages that models are sampled onto. Sampling more
# coarsely in age dramatically speeds up the code. Note, you'll need to
# regenerate the Cloudy emission line models if you change this.
log_width = 0.1  # 0.05

if not full_age_sampling:
    chosen_ages = np.arange(6., np.log10(cosmo.age(0.).value) + 9., log_width)
    chosen_age_lhs = make_bins(chosen_ages, make_rhs=True)[0]

    chosen_ages = 10**chosen_ages
    chosen_age_lhs = 10**chosen_age_lhs

    chosen_age_lhs[0] = 0.
    chosen_age_lhs[-1] = 10**9*cosmo.age(0.).value

    chosen_age_widths = chosen_age_lhs[1:] - chosen_age_lhs[:-1]


def make_dirs():
    """ Make local Bagpipes directory structure. """

    if not os.path.exists(working_dir + "/pipes"):
        os.mkdir(working_dir + "/pipes")

    if not os.path.exists(working_dir + "/pipes/plots"):
        os.mkdir(working_dir + "/pipes/plots")

    if not os.path.exists(working_dir + "/pipes/posterior"):
        os.mkdir(working_dir + "/pipes/posterior")

    if not os.path.exists(working_dir + "/pipes/cats"):
        os.mkdir(working_dir + "/pipes/cats")


def set_model_type(name):
    """ set up necessary variables for loading and manipulating models.
    Can be used to change between model types. """

    global model_type
    global chosen_ages
    global chosen_age_lhs
    global chosen_age_widths
    global chosen_live_frac

    # load is the function needed to get the attributes of the model set
    load = getattr(load_models, name)
    zmet_vals[name], gridwavs[name], ages[name], live_frac[name] = load()

    wavs_micron = np.copy(gridwavs[name])*10**-4

    age_lhs[name], age_widths[name] = make_bins(ages[name])

    age_lhs[name][0] = 0.
    age_widths[name][0] = age_lhs[name][1]

    model_type = name

    if full_age_sampling:
        chosen_ages = ages[model_type]
        chosen_age_lhs = age_lhs[model_type]
        chosen_age_widths = age_widths[model_type]

    if full_age_sampling:
        chosen_live_frac = live_frac[name][:, 1:].T

    else:
        chosen_live_frac = np.zeros((len(zmet_vals[name]),
                                     chosen_ages.shape[0]))

        for i in range(live_frac[name].shape[1]-1):
            chosen_live_frac[i, :] = np.interp(chosen_ages,
                                               10**live_frac[name][:, 0],
                                               live_frac[name][:, i + 1])

    # zmet_lims: edges of the metallicity bins.
    zmet_lims[model_type] = make_bins(zmet_vals[model_type], make_rhs=True)[0]

    zmet_lims[model_type][0] = 0.
    zmet_lims[model_type][-1] = 10.


def load_stellar_grid(zmet_val):
    """ Loads model grids at specified metallicity, compresses ages and
    resamples to chosen_modelgrid_wavs. """

    # If this raw grid is not already in allgrids from a previous model
    # object, load the file into that dictionary.
    zmet_ind = np.argmax(zmet_vals[model_type] == zmet_val)
    if model_type + str(zmet_ind) not in list(allstellargrids):
        get_grid = getattr(load_models, model_type + "_get_grid")
        allstellargrids[model_type + str(zmet_ind)] = get_grid(zmet_ind)

    grid = allstellargrids[model_type + str(zmet_ind)]

    # If requested, resample the ages of the models onto a uniform grid
    # in log10 of age (this is default behaviour).
    if not full_age_sampling:
        old_age_lhs = age_lhs[model_type]
        old_age_widths = age_widths[model_type]

        grid_compressed = np.zeros((chosen_age_widths.shape[0],
                                    grid.shape[1]))

        start = 0
        stop = 0

        # Calculate the spectra on the new age grid, loop over new bins
        for j in range(chosen_age_lhs.shape[0] - 1):

            # Find the first old age bin which is partially covered by
            # the new age bin
            while old_age_lhs[start + 1] <= chosen_age_lhs[j]:
                start += 1

            # Find the last old age bin which is partially covered by
            # the new age bin
            while old_age_lhs[stop+1] < chosen_age_lhs[j + 1]:
                stop += 1

            if stop == start:
                grid_compressed[j, :] = grid[start, :]

            else:
                start_fact = ((old_age_lhs[start + 1] - chosen_age_lhs[j])
                              / (old_age_lhs[start + 1] - old_age_lhs[start]))

                end_fact = ((chosen_age_lhs[j + 1] - old_age_lhs[stop])
                            / (old_age_lhs[stop + 1] - old_age_lhs[stop]))

                old_age_widths[start] *= start_fact
                old_age_widths[stop] *= end_fact

                width_slice = old_age_widths[start:stop + 1]
                summed = np.sum(np.expand_dims(width_slice, axis=1)
                                * grid[start:stop + 1, :], axis=0)

                grid_compressed[j, :] = summed/np.sum(width_slice)

                old_age_widths[start] /= start_fact
                old_age_widths[stop] /= end_fact

    # Otherwise pass the grid with its original age sampling
    elif full_age_sampling:
        grid_compressed = grid

    return grid_compressed


def load_cloudy_grid(zmet_val, logU):
    """ Loads Cloudy nebular continuum and emission lines at specified
    metallicity and logU. """

    zmet_ind = np.argmax(zmet_vals[model_type] == zmet_val)

    table_index = (zmet_vals[model_type].shape[0]*np.argmax(logU_grid == logU)
                   + zmet_ind + 1)

    key = str(zmet_val) + str(logU)

    path = install_dir + "/models/nebular/" + model_type + "/"
    linepath = path + model_type + "_nebular_line_grids.fits"
    contpath = path + model_type + "_nebular_cont_grids.fits"

    # If the raw grids are not already in allcloudy*grids from a
    # previous object, load the raw files into that dictionary.
    if key not in list(allcloudylinegrids):

        allcloudylinegrids[key] = fits.open(linepath)[table_index].data[1:, 1:]
        allcloudycontgrids[key] = fits.open(contpath)[table_index].data[1:, 1:]

    cloudy_cont_grid = allcloudycontgrids[key]
    cloudy_line_grid = allcloudylinegrids[key]

    # Check age sampling of the cloudy grid is the same as that for the
    # stellar grid, otherwise crash the code.
    cloudy_ages = fits.open(linepath)[1].data[1:, 0]

    for i in range(cloudy_ages.shape[0]):
        if not np.round(cloudy_ages[i], 5) == np.round(chosen_ages[i], 5):
            sys.exit("Bagpipes: Cloudy and stellar grids have different ages.")

    return cloudy_cont_grid, cloudy_line_grid

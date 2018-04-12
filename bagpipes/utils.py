from __future__ import print_function, division, absolute_import

import numpy as np
import os
import sys

from astropy.io import fits

install_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
working_dir = os.getcwd()

#sys.path.append(install_dir + "/bagpipes")

from .load_models import *

allstellargrids = {}
allcloudylinegrids = {}
allcloudycontgrids = {}

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70., Om0 = 0.3)
z_array = np.arange(0., 10., 0.01)
age_at_z = cosmo.age(z_array).value
ldist_at_z = cosmo.luminosity_distance(z_array).value

if not os.path.exists(install_dir + "/tables/igm/d_igm_grid_inoue14.fits"):
    from .igm_inoue2014 import make_table
    make_table()

D_IGM_grid = fits.open(install_dir + "/tables/igm/d_igm_grid_inoue14.fits")[1].data    
IGM_redshifts = np.arange(0.0, 10.01, 0.01)
IGM_wavs = np.arange(1.0, 1225.01, 1.0)

# Generate dictionaries containing information about spectral models and populate them using the set_model_type function.
gridwavs = {}
ages = {}
age_lhs = {}
age_widths = {}
zmet_vals = {}
live_mstar_frac = {}

model_type = "bc03_miles"
full_age_sampling = False
max_zred = 10.

logU_grid = np.arange(-4., -1.99, 0.5)

# Controls the grid of ages that models are sampled onto. Sampling more coarsely in age dramatically speeds up the code.
# Note, you'll need to regenerate the Cloudy emission line models if you change this.
log_width = 0.1

len_ages = int((10.21 - 6)/log_width) + 1

if not full_age_sampling:
    chosen_ages = 10.**(6. + log_width*np.arange(len_ages))
    chosen_age_lhs = np.zeros(len_ages+1)
    chosen_age_lhs[1:] = 10.**(6.0 + log_width/2. + log_width*np.arange(len_ages))
    chosen_age_widths = chosen_age_lhs[1:] - chosen_age_lhs[:-1]




def make_dirs():
    """ Make local Bagpipes directory structure. """
    if not os.path.exists(working_dir + "/pipes"):
        os.mkdir(working_dir + "/pipes")

    if not os.path.exists(working_dir + "/pipes/plots"):
        os.mkdir(working_dir + "/pipes/plots")

    if not os.path.exists(working_dir + "/pipes/pmn_chains"):
        os.mkdir(working_dir + "/pipes/pmn_chains")

    if not os.path.exists(working_dir + "/pipes/object_masks"):
        os.mkdir(working_dir + "/pipes/object_masks")

    if not os.path.exists(working_dir + "/pipes/filters"):
        os.mkdir(working_dir + "/pipes/filters")

    if not os.path.exists(working_dir + "/pipes/cats"):
        os.mkdir(working_dir + "/pipes/cats")


def make_bins(midpoints, make_rhs="False"):
    """ A function to take an array of bin midpoints and return an array of bin left hand side positions and widths. """

    bin_widths = np.zeros(midpoints.shape[0])

    if make_rhs == "True":
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



def set_model_type(chosen_models):
    """ set up necessary variables for loading and manipulating spectral  Can be used to change between model types. """
    global model_type
    global chosen_ages
    global chosen_age_lhs
    global chosen_age_widths
    global chosen_live_mstar_frac

    zmet_vals[chosen_models], gridwavs[chosen_models], ages[chosen_models], live_mstar_frac[chosen_models] = globals()[chosen_models]()

    wavs_micron = np.copy(gridwavs[chosen_models])*10**-4

    age_lhs[chosen_models], age_widths[chosen_models] = make_bins(ages[chosen_models])

    age_lhs[chosen_models][0] = 0.
    age_widths[chosen_models][0] = age_lhs[chosen_models][1]

    model_type = chosen_models

    if full_age_sampling == True:
        chosen_ages = ages[model_type]
        chosen_age_lhs = age_lhs[model_type]
        chosen_age_widths = age_widths[model_type]

    if full_age_sampling == False:

        chosen_live_mstar_frac = np.zeros((len(zmet_vals[chosen_models]), chosen_ages.shape[0]))

        for i in range(live_mstar_frac[chosen_models].shape[1]-1):
            chosen_live_mstar_frac[i,:] = np.interp(chosen_ages, 10**live_mstar_frac[chosen_models][:,0], live_mstar_frac[chosen_models][:,i+1])



""" Loads model grids at specified metallicity, compresses ages and resamples to chosen_modelgrid_wavs. """
def load_stellar_grid(zmet_ind):

    # If this raw grid is not already in allgrids from a previous model object, load the file into that dictionary
    if model_type + str(zmet_ind) not in list(allstellargrids):
        allstellargrids[model_type + str(zmet_ind)] = globals()[model_type + "_get_grid"](zmet_ind)

    grid = allstellargrids[model_type + str(zmet_ind)]

    # If requested, resample the ages of the models onto a uniform grid (this is default behaviour)
    if full_age_sampling == False:
        old_age_lhs = age_lhs[model_type]
        old_age_widths = age_widths[model_type]

        grid_compressed = np.zeros((chosen_age_widths.shape[0], grid.shape[1]))

        start = 0
        stop = 0

        # Calculate the spectra on the new age grid, loop over the new bins
        for j in range(chosen_age_lhs.shape[0]-1):

            # Find the first old age bin which is partially covered by the new age bin
            while old_age_lhs[start+1] <= chosen_age_lhs[j]:
                start += 1

            # Find the last old age bin which is partially covered by the new age bin
            while old_age_lhs[stop+1] < chosen_age_lhs[j+1]:
                stop += 1

            if stop == start:
                grid_compressed[j, :] = grid[start, :]

            else:
                start_factor = (old_age_lhs[start+1] - chosen_age_lhs[j])/(old_age_lhs[start+1] - old_age_lhs[start])
                end_factor = (chosen_age_lhs[j+1] - old_age_lhs[stop])/(old_age_lhs[stop+1] - old_age_lhs[stop])

                old_age_widths[start] *= start_factor
                old_age_widths[stop] *= end_factor

                grid_compressed[j, :] = np.sum(np.expand_dims(old_age_widths[start:stop+1], axis=1)*grid[start:stop+1, :], axis=0)/np.sum(old_age_widths[start:stop+1])

                old_age_widths[start] /= start_factor
                old_age_widths[stop] /= end_factor

    # Otherwise pass the grid with its original age sampling
    elif full_age_sampling == True:
        grid_compressed = grid

    return grid_compressed



""" Loads Cloudy nebular continuum and emission lines at specified metallicity and logU. """
def load_cloudy_grid(zmet_ind, logU):

    table_index = zmet_vals[model_type].shape[0]*np.argmax(logU_grid==logU) + zmet_ind + 1

    # If the raw grids are not already in allcloudy*grids from a previous object, load the raw files into that dictionary
    if str(zmet_ind) + str(logU) not in list(allcloudylinegrids):
        allcloudylinegrids[str(zmet_ind) + str(logU)] = fits.open(install_dir + "/tables/nebular/" + model_type + "/" + model_type + "_nebular_line_grids.fits")[table_index].data[1:,1:]
        allcloudycontgrids[str(zmet_ind) + str(logU)] = fits.open(install_dir + "/tables/nebular/" + model_type + "/" + model_type + "_nebular_cont_grids.fits")[table_index].data[1:,1:]

    cloudy_cont_grid = allcloudycontgrids[str(zmet_ind) + str(logU)]
    cloudy_line_grid = allcloudylinegrids[str(zmet_ind) + str(logU)]

    #Check age sampling of the cloudy grid is the same as that for the stellar grid, otherwise crash the code
    cloudy_ages = fits.open(install_dir + "/tables/nebular/" + model_type + "/" + model_type + "_nebular_line_grids.fits")[1].data[1:,0]

    for i in range(cloudy_ages.shape[0]):
        if not np.round(cloudy_ages[i], 5) == np.round(chosen_ages[i], 5):
            sys.exit("BAGPIPES: Cloudy grid has different ages to stellar model age grid.")

    return cloudy_cont_grid, cloudy_line_grid


set_model_type(model_type)





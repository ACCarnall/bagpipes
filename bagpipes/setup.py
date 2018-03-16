""" Contains code for general Bagpipes setting up and configuration. """

import numpy as np
import os

# Set up path variables for use elsewhere in the code. 
install_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
working_dir = os.getcwd()

import load_models


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



def set_model_type(models):
    """ set up necessary variables for loading and manipulating spectral models. Can be used to change between model types. """

    zmet_fnames[models], zmet_vals[models], len_wavs[models], gridwavs[models], ages[models], mstar_liv[models] = getattr(load_models, models)()

    wavs_micron = np.copy(gridwavs[models])*10**-4

    age_lhs[models], age_widths[models] = make_bins(ages[models])

    age_lhs[models][0] = 0.
    age_widths[models][0] = age_lhs[models][1]

    global model_type
    model_type = models

    if full_age_sampling == True:
        global chosen_ages
        global chosen_age_lhs
        global chosen_age_widths
        chosen_ages = ages[model_type]
        chosen_age_lhs = age_lhs[model_type]
        chosen_age_widths = age_widths[model_type]

    if full_age_sampling == False:
        global chosen_mstar_liv

        chosen_mstar_liv = np.zeros((len(zmet_vals[models]), chosen_ages.shape[0]))

        for i in range(mstar_liv[models].shape[1]-1):
            chosen_mstar_liv[i,:] = np.interp(chosen_ages, 10**mstar_liv[models][:,0], mstar_liv[models][:,i+1])



# Make local Bagpipes directory structure.
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


# Generate dictionaries containing information about spectral models and populate them using the set_model_type function.
gridwavs = {}
ages = {}
age_lhs = {}
age_widths = {}
zmet_vals = {}
len_wavs = {}
zmet_fnames = {} 
mstar_liv = {}

model_type = "bc03_miles"
full_age_sampling = False
max_zred = 10.

logU_grid = np.arange(-4., -0.99, 0.5)


log_width = 0.05 # Controls the grid of ages that models are sampled onto. Sampling more coarsely in age dramatically speeds up the code.

len_ages = int((10.21 - 6)/log_width) + 1

if full_age_sampling == False:
    chosen_ages = 10.**(6. + log_width*np.arange(len_ages))
    chosen_age_lhs = np.zeros(len_ages+1)
    chosen_age_lhs[1:] = 10.**(6.0 + log_width/2. + log_width*np.arange(len_ages))
    chosen_age_widths = chosen_age_lhs[1:] - chosen_age_lhs[:-1]

set_model_type(model_type)


import numpy as np
import os

install_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."

import load_models

def make_bins(wavelengths, make_rhs="False"):
    bin_widths = np.zeros(wavelengths.shape[0])

    if make_rhs == "True":
        bin_lhs = np.zeros(wavelengths.shape[0]+1)
        bin_lhs[0] = wavelengths[0] - (wavelengths[1]-wavelengths[0])/2
        bin_widths[-1] = (wavelengths[-1] - wavelengths[-2])
        bin_lhs[-1] = wavelengths[-1] + (wavelengths[-1]-wavelengths[-2])/2
        bin_lhs[1:-1] = (wavelengths[1:] + wavelengths[:-1])/2
        bin_widths[:-1] = bin_lhs[1:-1]-bin_lhs[:-2]

    else:
        bin_lhs = np.zeros(wavelengths.shape[0])
        bin_lhs[0] = wavelengths[0] - (wavelengths[1]-wavelengths[0])/2
        bin_widths[-1] = (wavelengths[-1] - wavelengths[-2])
        bin_lhs[1:] = (wavelengths[1:] + wavelengths[:-1])/2
        bin_widths[:-1] = bin_lhs[1:]-bin_lhs[:-1]

    return bin_lhs, bin_widths


#set up necessary variables for loading and manipulating spectral models:


def set_model_type(models):

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

        chosen_mstar_liv = np.zeros((chosen_ages.shape[0], len(zmet_vals[models])))

        for i in range(mstar_liv[models].shape[1]-1):
            chosen_mstar_liv[:,i] = np.interp(chosen_ages, 10**mstar_liv[models][:,0], mstar_liv[models][:,i+1])



gridwavs = {}
ages = {}
age_lhs = {}
age_widths = {}
zmet_vals = {}
len_wavs = {}
zmet_fnames = {} 
mstar_liv = {}

model_type = "bc03_miles"
n_live_points = 400
sampling_efficiency = "parameter"
pmn_verbose = True
full_age_sampling = False
max_zred = 10.

logU_grid = np.arange(-4., -0.99, 0.5)

#Controls the grid of ages that models are sampled onto. Sampling more coarsely in age dramatically speeds up the code.

log_width = 0.05

len_ages = int((10.21 - 6)/log_width) + 1

if full_age_sampling == False:
    chosen_ages = 10.**(6. + log_width*np.arange(len_ages))
    chosen_age_lhs = np.zeros(len_ages+1)
    chosen_age_lhs[1:] = 10.**(6.0 + log_width/2. + log_width*np.arange(len_ages))
    chosen_age_widths = chosen_age_lhs[1:] - chosen_age_lhs[:-1]

set_model_type(model_type)


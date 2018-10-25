from __future__ import print_function, division, absolute_import

import numpy as np

from .. import utils
from .. import config


class chemical_enrichment_history(object):

    def __init__(self, model_comp, sfh_weights):

        self.zmet_vals = config.metallicities
        self.zmet_lims = config.metallicity_bins

        self.grid_comp = {}
        self.grid = np.zeros((self.zmet_vals.shape[0],
                              config.age_sampling.shape[0]))

        for comp in list(sfh_weights):
            if comp is not "total":
                self.grid_comp[comp] = self.delta(model_comp[comp],
                                                  sfh_weights[comp])

                self.grid += self.grid_comp[comp]

    def delta(self, comp, sfh):
        """ Delta function metallicity history. """

        zmet = comp["metallicity"]

        weights = np.zeros(self.zmet_vals.shape[0])

        high_ind = self.zmet_vals[self.zmet_vals < zmet].shape[0]

        if high_ind == self.zmet_vals.shape[0]:
            weights[-1] = 1.

        elif high_ind == 0:
            weights[0] = 1.

        else:
            low_ind = high_ind - 1
            width = (self.zmet_vals[high_ind] - self.zmet_vals[low_ind])
            weights[high_ind] = (zmet - self.zmet_vals[low_ind])/width
            weights[high_ind-1] = 1 - weights[high_ind]

        return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)

    def exp(self, comp, sfh):
        """ P(Z) = exp(-z/z_mean). Currently no age dependency! """

        mean_zmet = comp["metallicity"]

        weights = np.zeros(self.zmet_vals.shape[0])

        vals_hr = np.arange(0., 10., 0.01) + 0.005

        factors_hr = (1./mean_zmet)*np.exp(-vals_hr/mean_zmet)

        for i in range(weights.shape[0]):
            lowmask = (vals_hr > self.zmet_lims[i])
            highmask = (vals_hr < self.zmet_lims[i+1])
            weights[i] = np.sum(0.01*factors_hr[lowmask & highmask])

        return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)

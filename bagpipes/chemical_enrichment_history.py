from __future__ import print_function, division, absolute_import

import numpy as np

from . import utils


class chemical_enrichment_history:

    def __init__(self, model_comp):

        self.model_comp = model_comp

        self.zmet_vals = utils.zmet_vals[utils.model_type]
        self.zmet_lims = utils.zmet_lims[utils.model_type]

        self.zmet_weights = {}

        for name in list(self.model_comp):
            if (not name.startswith(("dust", "nebular", "polynomial", "noise"))
                    and isinstance(self.model_comp[name], dict)):

                    comp = self.model_comp[name]
                    if (("metallicity_dist" in list(comp))
                            and comp["metallicity_dist"]):

                        self.zmet_weights[name] = self.exp(comp)

                    else:
                        self.zmet_weights[name] = self.delta(comp)

    def delta(self, comp):
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

        return weights

    def exp(self, comp):
        """ P(Z) = exp(-z/z_mean). Currently no age dependency! """

        mean_zmet = comp["metallicity"]

        weights = np.zeros(self.zmet_vals.shape[0])

        vals_hr = np.arange(0., 10., 0.01) + 0.005

        factors_hr = (1./mean_zmet)*np.exp(-vals_hr/mean_zmet)

        for i in range(weights.shape[0]):
            lowmask = (vals_hr > self.zmet_lims[i])
            highmask = (vals_hr < self.zmet_lims[i+1])
            weights[i] = np.sum(0.01*factors_hr[lowmask & highmask])

        return weights

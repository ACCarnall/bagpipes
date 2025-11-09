from __future__ import print_function, division, absolute_import

import numpy as np

from .. import config


class chemical_enrichment_history(object):

    def __init__(self, model_comp, sfh_weights):

        self.zmet_vals = config.metallicities
        self.zmet_lims = config.metallicity_bins

        self.grid_comp = {}
        self.grid = np.zeros((self.zmet_vals.shape[0],
                              config.age_sampling.shape[0]))

        for comp in list(sfh_weights):
            if comp == "total":
                continue

            zmet_keys = ['metallicity_type', 'metallicity_scatter']

            if all(key not in model_comp[comp].keys() for key in zmet_keys):
                self.grid_comp[comp] = self.delta(model_comp[comp],
                                                  sfh_weights[comp])

            elif 'metallicity_type' in model_comp[comp].keys():
                zmet_key = "metallicity_type"
                self.grid_comp[comp] = getattr(self, model_comp[comp][zmet_key]
                                               )(model_comp[comp],
                                                 sfh_weights[comp])

            else:
                zmet_key = "metallicity_scatter"
                self.grid_comp[comp] = getattr(self, model_comp[comp][zmet_key]
                                               )(model_comp[comp],
                                                 sfh_weights[comp])

            self.grid += self.grid_comp[comp]

    def delta(self, comp, sfh, zmet=None, nested=False):
        """ Delta function metallicity history, default unless specified. """

        if zmet is None:
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

        if nested:
            return weights
        else:
            return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)

    def metallicity_bins(self, comp, sfh):
        """ Different metallicities in each specificed time bin. """

        bin_edges = np.array(comp["bin_edges"])[::-1]*10**6
        n_bins = len(bin_edges) - 1
        ages = config.age_sampling
        grid = np.zeros((self.zmet_vals.shape[0], sfh.shape[0]))

        for i in range(1, n_bins+1):
            zmet = comp["metallicity" + str(i)]

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

            mask = (ages < bin_edges[i-1]) & (ages > bin_edges[i])
            grid[:, mask] = np.expand_dims(weights, axis=1)

        return grid*sfh

    def metallicity_bins_continuity(self, comp, sfh):
        """ Work like Leja continuity SFH, zmet varies with dirichlet prior. """

        bin_edges = np.array(comp["bin_edges"])[::-1]*10**6
        n_bins = len(bin_edges) - 1
        ages = config.age_sampling
        grid = np.zeros((self.zmet_vals.shape[0], sfh.shape[0]))
        dzmet = [comp["dzmet" + str(i)] for i in range(1, n_bins)]
        for i in range(1, n_bins+1):

            zmet = comp["metallicity1"]
            if i >= 2:
                zmet = 10**(np.log10(comp["metallicity1"])
                            + np.sum(dzmet[:i-1]))

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

            mask = (ages < bin_edges[i-1]) & (ages > bin_edges[i])
            grid[:, mask] = np.expand_dims(weights, axis=1)

        return grid*sfh

    def exp(self, comp, sfh, zmet=None, nested=False):
        """ zmet distributed like P(Z) = exp(-z/z_mean), no age dependency. """

        if zmet is None:
            mean_zmet = comp["metallicity"]
        else:
            mean_zmet = zmet

        weights = np.zeros(self.zmet_vals.shape[0])

        vals_hr = np.arange(0., 10., 0.01) + 0.005

        factors_hr = (1./mean_zmet)*np.exp(-vals_hr/mean_zmet)

        for i in range(weights.shape[0]):
            lowmask = (vals_hr > self.zmet_lims[i])
            highmask = (vals_hr < self.zmet_lims[i+1])
            weights[i] = np.sum(0.01*factors_hr[lowmask & highmask])

        if nested:
            return weights
        else:
            return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)

    def lognorm(self, comp, sfh, zmet=None, nested=False):
        """
        log normal metallicity distribution for coeval stars.
        Functional form:
        P(x) = 1/(x*sigma*np.sqrt(2*np.pi))
               * np.exp(-(np.log(x)-mu)**2/(2*sigma**2))

        where mu = ln(metallicity mean),
              sigma = some concentration measurement
        """

        if zmet is None:
            log_mean_zmet = np.log(comp["metallicity"])
        else:
            log_mean_zmet = np.log(zmet)
        sigma = 0.45

        weights = np.zeros(self.zmet_vals.shape[0])

        vals_hr = np.arange(0., 10., 0.01) + 0.005

        factors_hr = (1/(vals_hr*sigma*np.sqrt(2*np.pi))
                      * np.exp(-(np.log(vals_hr)-log_mean_zmet)**2
                               / (2*sigma**2)))

        for i in range(weights.shape[0]):
            lowmask = (vals_hr > self.zmet_lims[i])
            highmask = (vals_hr < self.zmet_lims[i+1])
            weights[i] = np.sum(0.01*factors_hr[lowmask & highmask])

        if nested:
            return weights
        else:
            return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)

    def constant(self, comp, sfh):
        """ constant metallicity without any variation in time, distribution
        of coeval stars can be specified thorugh 'metallicity_scatter'
        """

        zmet = comp["metallicity"]
        if "metallicity_scatter" not in comp.keys():
            comp["metallicity_scatter"] = "delta"

        weights = getattr(self, comp['metallicity_scatter']
                          )(comp, sfh, zmet=zmet, nested=True)
        return np.expand_dims(weights, axis=1)*np.expand_dims(sfh, axis=0)

    def two_step(self, comp, sfh):
        """ 2-step metallicities (time-varying!)
        time of shift is a free parameter """

        zmet_old = comp["metallicity_old"]
        zmet_young = comp["metallicity_young"]
        step_age = comp["metallicity_step_age"]*10**9
        if "metallicity_scatter" not in comp.keys():
            comp["metallicity_scatter"] = 'delta'

        # get SSP ages
        SSP_ages = config.age_sampling
        SSP_age_bins = config.age_bins

        # loop through all SSP ages
        zmet_comp = np.zeros((self.zmet_vals.shape[0], sfh.shape[0]))
        for i, agei in enumerate(SSP_ages):
            # Check SSP age higher boundary>step_age, lower boundary<step_age
            if SSP_age_bins[i+1] > step_age and SSP_age_bins[i] < step_age:
                # interp between to get metallicity at this SSP
                width = SSP_age_bins[i+1] - SSP_age_bins[i]
                old_weight = (SSP_age_bins[i+1] - step_age)/width
                burst_weight = (step_age - SSP_age_bins[i])/width
                SSP_zmet = old_weight*zmet_old + burst_weight*zmet_young
                # weights from metallicity scatter
                zmet_comp[:, i] = getattr(self, comp['metallicity_scatter']
                                          )(comp, sfh, zmet=SSP_zmet,
                                            nested=True)

            # if before step_age
            elif SSP_age_bins[i] > step_age:
                # weights from metallicity scatter
                zmet_comp[:, i] = getattr(self, comp['metallicity_scatter']
                                          )(comp, sfh, zmet=zmet_old,
                                            nested=True)

            # if after step_age
            elif SSP_age_bins[i+1] < step_age:
                # weights from metallicity scatter
                zmet_comp[:, i] = getattr(self, comp['metallicity_scatter']
                                          )(comp, sfh, zmet=zmet_young,
                                            nested=True)

        return zmet_comp*np.expand_dims(sfh, axis=0)

    def psb_two_step(self, comp, sfh):
        """
        2-step metallicities (time-varying!) for psb_wild2020 SFH model
        shift in metallicity at burstage parameter from SFH model
        For details, see Leung et al. 2024
        (https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.4029L)
        """

        zmet_old = comp["metallicity_old"]
        zmet_burst = comp["metallicity_burst"]
        burstage = comp["burstage"]*10**9
        if "metallicity_scatter" not in comp.keys():
            comp["metallicity_scatter"] = 'delta'

        # get SSP ages
        SSP_ages = config.age_sampling
        SSP_age_bins = config.age_bins

        # loop through all SSP ages
        zmet_comp = np.zeros((self.zmet_vals.shape[0], sfh.shape[0]))
        for i, agei in enumerate(SSP_ages):
            # Check SSP age higher boundary>tburst, lower boundary<tburst
            if SSP_age_bins[i+1] > burstage and SSP_age_bins[i] < burstage:
                # interp between to get metallicity at this SSP
                width = SSP_age_bins[i+1] - SSP_age_bins[i]
                old_weight = (SSP_age_bins[i+1] - burstage)/width
                burst_weight = (burstage - SSP_age_bins[i])/width
                SSP_zmet = old_weight*zmet_old + burst_weight*zmet_burst
                # weights from metallicity scatter
                zmet_comp[:, i] = getattr(self, comp['metallicity_scatter']
                                          )(comp, sfh, zmet=SSP_zmet,
                                            nested=True)

            # if before tburst
            elif SSP_age_bins[i] > burstage:
                # weights from metallicity scatter
                zmet_comp[:, i] = getattr(self, comp['metallicity_scatter']
                                          )(comp, sfh, zmet=zmet_old,
                                            nested=True)

            # if after tburst
            elif SSP_age_bins[i+1] < burstage:
                # weights from metallicity scatter
                zmet_comp[:, i] = getattr(self, comp['metallicity_scatter']
                                          )(comp, sfh, zmet=zmet_burst,
                                            nested=True)

        return zmet_comp*np.expand_dims(sfh, axis=0)

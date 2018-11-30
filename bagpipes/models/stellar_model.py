from __future__ import print_function, division, absolute_import

import numpy as np

from .. import config
from .. import utils


class stellar(object):
    """ Allows access to and maniuplation of stellar emission models.

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values desired for the stellar models.
    """

    def __init__(self, wavelengths):
        self.wavelengths = wavelengths

        # Resample the grid in wavelength and then in age.
        grid_raw_ages = self._resample_in_wavelength()
        self.grid = self._resample_in_age(grid_raw_ages)

    def _resample_in_wavelength(self):
        """ Resamples the raw stellar grids to the input wavs. """

        grid_raw_ages = np.zeros((self.wavelengths.shape[0],
                                  config.metallicities.shape[0],
                                  config.raw_stellar_ages.shape[0]))

        for i in range(config.metallicities.shape[0]):
            for j in range(config.raw_stellar_ages.shape[0]):

                raw_grid = config.raw_stellar_grid[i].data
                grid_raw_ages[:, i, j] = np.interp(self.wavelengths,
                                                   config.wavelengths,
                                                   raw_grid[j, :],
                                                   left=0., right=0.)

        return grid_raw_ages

    def _resample_in_age(self, grid_raw_ages):
        """ Resamples the intermediate stellar grids to the age sampling
        which is set in the config file. This has to be done carefully
        as summing the contributions from different ages in the correct
        ratios is very important for obtaining realistic results. """

        grid = np.zeros((self.wavelengths.shape[0],
                         config.metallicities.shape[0],
                         config.age_sampling.shape[0]))

        raw_age_lhs, raw_age_widths = utils.make_bins(config.raw_stellar_ages,
                                                      make_rhs=True)

        # Force raw ages to span full range from 0 to age of Universe.
        raw_age_widths[0] += raw_age_lhs[0]
        raw_age_lhs[0] = 0.

        if raw_age_lhs[-1] < config.age_bins[-1]:
            raw_age_widths[-1] += config.age_bins[-1] - raw_age_lhs[-1]
            raw_age_lhs[-1] = config.age_bins[-1]

        start = 0
        stop = 0

        # Loop over the new age bins
        for j in range(config.age_bins.shape[0] - 1):

            # Find the first raw bin partially covered by the new bin
            while raw_age_lhs[start + 1] <= config.age_bins[j]:
                start += 1

            # Find the last raw bin partially covered by the new bin
            while raw_age_lhs[stop+1] < config.age_bins[j + 1]:
                stop += 1

            # If new bin falls completely within one raw bin
            if stop == start:
                grid[:, :, j] = grid_raw_ages[:, :, start]

            # If new bin has contributions from more than one raw bin
            else:
                start_fact = ((raw_age_lhs[start + 1] - config.age_bins[j])
                              / (raw_age_lhs[start + 1] - raw_age_lhs[start]))

                end_fact = ((config.age_bins[j + 1] - raw_age_lhs[stop])
                            / (raw_age_lhs[stop + 1] - raw_age_lhs[stop]))

                raw_age_widths[start] *= start_fact
                raw_age_widths[stop] *= end_fact

                width_slice = raw_age_widths[start:stop + 1]

                summed = np.sum(np.expand_dims(width_slice, axis=0)
                                * grid_raw_ages[:, :, start:stop + 1], axis=2)

                grid[:, :, j] = summed/np.sum(width_slice)

                raw_age_widths[start] /= start_fact
                raw_age_widths[stop] /= end_fact

        return grid

    def spectrum(self, sfh_ceh, t_bc=0.):
        """ Obtain a split 1D spectrum for a given star-formation and
        chemical enrichment history, one for ages lower than t_bc, one
        for ages higher than t_bc. This allows extra dust to be applied
        to the younger population still within its birth clouds.

        parameters
        ----------

        sfh_ceh : numpy.ndarray
            2D array containing the desired star-formation and
            chemical evolution history.

        t_bc : float
            The age at which to split the spectrum in Gyr.
        """

        t_bc *= 10**9
        spectrum_young = np.zeros_like(self.wavelengths)
        spectrum = np.zeros_like(self.wavelengths)

        index = config.age_bins[config.age_bins < t_bc].shape[0]
        old_weight = (config.age_bins[index] - t_bc)/config.age_widths[index-1]

        if index == 0:
            index += 1

        for i in range(config.metallicities.shape[0]):
            if sfh_ceh[i, :index].sum() > 0.:
                sfh_ceh[:, index-1] *= (1. - old_weight)

                spectrum_young += np.sum(self.grid[:, i, :index]
                                         * sfh_ceh[i, :index], axis=1)

                sfh_ceh[:, index-1] /= (1. - old_weight)

            if sfh_ceh[i, index-1:].sum() > 0.:
                sfh_ceh[:, index-1] *= old_weight

                spectrum += np.sum(self.grid[:, i, index-1:]
                                   * sfh_ceh[i, index-1:], axis=1)

                sfh_ceh[:, index-1] /= old_weight

        if t_bc == 0.:
            return spectrum

        return spectrum_young, spectrum

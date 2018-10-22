from __future__ import print_function, division, absolute_import

import numpy as np

from .. import config


class igm(object):
    """ Allows access to and maniuplation of the IGM attenuation models
    of Inoue (2014).

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values desired for the DL07 models.
    """

    def __init__(self, wavelengths):
        self.wavelengths = wavelengths

        self.grid = self._resample_in_wavelength()

    def _resample_in_wavelength(self):
        """ Resample the raw grid to the input wavelengths. """

        grid = np.zeros((self.wavelengths.shape[0],
                         config.igm_redshifts.shape[0]))

        for i in range(config.igm_redshifts.shape[0]):
            grid[:, i] = np.interp(self.wavelengths, config.igm_wavelengths,
                                   config.raw_igm_grid[i, :],
                                   left=0., right=1.)

        return grid

    def trans(self, redshift):
        """ Get the IGM transmission at a given redshift. """

        redshift_mask = (config.igm_redshifts < redshift)
        zred_ind = config.igm_redshifts[redshift_mask].shape[0]

        zred_fact = ((redshift - config.igm_redshifts[zred_ind-1])
                     / (config.igm_redshifts[zred_ind]
                        - config.igm_redshifts[zred_ind-1]))

        if zred_ind == 0:
            zred_ind += 1
            zred_fact = 0.

        weights = np.array([1. - zred_fact, zred_fact])
        igm_trans = np.sum(weights*self.grid[:, zred_ind-1:zred_ind+1], axis=1)

        return igm_trans

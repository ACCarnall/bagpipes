from __future__ import print_function, division, absolute_import

import numpy as np

from .. import config


class dust_emission(object):
    """ Allows access to and maniuplation of the dust emission models
    of Draine + Li (2007). Currently very simple, possibly could be sped
    up in some circumstances by pre-interpolating to the input wavs.

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values desired for the DL07 models.
    """

    def __init__(self, wavelengths):
        self.wavelengths = wavelengths

    def spectrum(self, qpah, umin, gamma):
        """ Get the 1D spectrum for a given set of model parameters. """

        qpah_ind = config.qpah_vals[config.qpah_vals < qpah].shape[0]
        umin_ind = config.umin_vals[config.umin_vals < umin].shape[0]

        qpah_fact = ((qpah - config.qpah_vals[qpah_ind-1])
                     / (config.qpah_vals[qpah_ind]
                        - config.qpah_vals[qpah_ind-1]))

        umin_fact = ((umin - config.umin_vals[umin_ind-1])
                     / (config.umin_vals[umin_ind]
                        - config.umin_vals[umin_ind-1]))

        umin_w = np.array([(1 - umin_fact), umin_fact])

        lqpah_only = config.dust_grid_umin_only[qpah_ind].data
        hqpah_only = config.dust_grid_umin_only[qpah_ind+1].data
        tqpah_only = (qpah_fact*hqpah_only[:, umin_ind:umin_ind+2]
                      + (1-qpah_fact)*lqpah_only[:, umin_ind:umin_ind+2])

        lqpah_umax = config.dust_grid_umin_umax[qpah_ind].data
        hqpah_umax = config.dust_grid_umin_umax[qpah_ind+1].data
        tqpah_umax = (qpah_fact*hqpah_umax[:, umin_ind:umin_ind+2]
                      + (1-qpah_fact)*lqpah_umax[:, umin_ind:umin_ind+2])

        interp_only = np.sum(umin_w*tqpah_only, axis=1)
        interp_umax = np.sum(umin_w*tqpah_umax, axis=1)

        model = gamma*interp_umax + (1 - gamma)*interp_only

        spectrum = np.interp(self.wavelengths,
                             config.dust_grid_umin_only[1].data[:, 0],
                             model, left=0., right=0.)

        return spectrum

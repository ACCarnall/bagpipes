from __future__ import print_function, division, absolute_import

import numpy as np


class agn(object):
    """ A basic rest-optical AGN continuum + broad line model.

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values for the continuum.

    param : dict
        Parameters for the AGN model.
    """

    def __init__(self, wavelengths):
        self.wavelengths = wavelengths

    def update(self, param):
        self.param = param

        tau = 5000.
        mask1 = (self.wavelengths < 5000.)
        mask2 = np.invert(mask1)

        agn_spec = np.zeros_like(self.wavelengths)

        agn_spec[mask1] = self.wavelengths[mask1] ** param["alphalam"]
        agn_spec[mask2] = self.wavelengths[mask2] ** param["betalam"]

        agn_spec[mask1] /= agn_spec[mask1][-1]
        agn_spec[mask2] /= agn_spec[mask2][0]

        agn_spec /= agn_spec[np.argmin(np.abs(self.wavelengths - 5100.))]
        agn_spec *= param["f5100A"]

        agn_spec += self.gaussian_model(4861.35, param["sigma"],
                                        param["hanorm"]/2.86)

        agn_spec += self.gaussian_model(6562.80, param["sigma"],
                                        param["hanorm"])

        self.spectrum = agn_spec

    def gaussian_model(self, central_wav, sigma_vel, norm):
        x = self.wavelengths
        sigma_aa = sigma_vel*central_wav/(3*10**5)
        gauss = (norm/(sigma_aa*np.sqrt(2*np.pi)))
        gauss *= np.exp(-0.5*(x-central_wav)**2/sigma_aa**2)

        return gauss

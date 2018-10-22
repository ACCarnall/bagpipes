from __future__ import print_function, division, absolute_import

import numpy as np

from .. import config


class dust_attenuation(object):
    """ Allows access to and maniuplation of dust attenuation models.
    Three options are currently implemented as described in Carnall et
    al. (2018): Calzetti et al. (2000), Cardelli et al. (1989) and a
    model based on Charlot + Fall (2000).

    Parameters
    ----------

    wavelengths : np.ndarray
        1D array of wavelength values for the continuum.

    type : str
        The type of dust model, either "Calzetti", "Cardelli" or "CF00".
    """

    def __init__(self, wavelengths, type="Calzetti"):
        self.wavelengths = wavelengths
        self.type = type

        # Calculate A(lambda)/A(V) for the continuum and emission lines
        if type == "Calzetti":
            self.A_continuum = self._calzetti_attenuation(wavelengths)
            self.A_lines = self._calzetti_attenuation(config.line_wavs)

        if type == "Cardelli":
            self.A_continuum = self._cardelli_extinction(wavelengths)
            self.A_lines = self._cardelli_extinction(config.line_wavs)

        if type == "CF00":
            self.A_continuum = self._cf00_attenuation(wavelengths)
            self.A_lines = self._cf00_attenuation(config.line_wavs)

    def trans(self, Av, n=1):
        """ Return transmission as a function of wavelength for the
        continuum wavelengths as a function of Av and (maybe) n. """

        return 10**(-Av*self.A_continuum**n/2.5)

    def line_trans(self, Av, n=1):
        """ Return transmission as a function of wavelength for the
        emission line wavelengths as a function of Av and (maybe) n. """

        return 10**(-Av*self.A_lines**n/2.5)

    def _cardelli_extinction(self, wavs):
        """ Calculate the ratio A(lambda)/A(V) for the Cardelli et al.
        (1989) extinction curve. """

        A_lambda = np.zeros_like(wavs)

        inv_mic = 1./(wavs*10.**-4.)

        mask1 = (inv_mic < 1.1)
        mask2 = (inv_mic >= 1.1) & (inv_mic < 3.3)
        mask3 = (inv_mic >= 3.3) & (inv_mic < 5.9)
        mask4 = (inv_mic >= 5.9) & (inv_mic < 8.)
        mask5 = (inv_mic >= 8.)

        A_lambda[mask1] = (0.574*inv_mic[mask1]**1.61
                           + (-0.527*inv_mic[mask1]**1.61)/3.1)

        y = inv_mic[mask2] - 1.82

        A_lambda[mask2] = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3
                           + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6
                           + 0.32999*y**7 + (1.41388*y + 2.28305*y**2
                                             + 1.07233*y**3 - 5.38434*y**4
                                             - 0.62251*y**5 + 5.30260*y**6
                                             - 2.09002*y**7)/3.1)

        A_lambda[mask3] = ((1.752 - 0.316*inv_mic[mask3]
                            - (0.104)/((inv_mic[mask3]-4.67)**2 + 0.341))
                           + (-3.09 + 1.825*inv_mic[mask3]
                              + 1.206/((inv_mic[mask3]-4.62)**2 + 0.263))/3.1)

        A_lambda[mask4] = ((1.752 - 0.316*inv_mic[mask4]
                            - (0.104)/((inv_mic[mask4]-4.67)**2 + 0.341)
                            - 0.04473*(inv_mic[mask4] - 5.9)**2
                            - 0.009779*(inv_mic[mask4]-5.9)**3)
                           + (-3.09 + 1.825*inv_mic[mask4]
                              + 1.206/((inv_mic[mask4]-4.62)**2 + 0.263)
                              + 0.2130*(inv_mic[mask4]-5.9)**2
                              + 0.1207*(inv_mic[mask4]-5.9)**3)/3.1)

        A_lambda[mask5] = ((-1.073
                            - 0.628*(inv_mic[mask5] - 8.)
                            + 0.137*(inv_mic[mask5] - 8.)**2
                            - 0.070*(inv_mic[mask5] - 8.)**3)
                           + (13.670
                              + 4.257*(inv_mic[mask5] - 8.)
                              - 0.420*(inv_mic[mask5] - 8.)**2
                              + 0.374*(inv_mic[mask5] - 8.)**3)/3.1)

        return A_lambda

    def _calzetti_attenuation(self, wavs):
        """ Calculate the ratio A(lambda)/A(V) for the Calzetti et al.
        (2000) attenuation curve. """

        A_lambda = np.zeros_like(wavs)

        wavs_mic = wavs*10**-4

        mask1 = (wavs < 1200.)
        mask2 = (wavs < 6300.) & (wavs >= 1200.)
        mask3 = (wavs < 31000.) & (wavs >= 6300.)

        A_lambda[mask1] = ((wavs_mic[mask1]/0.12)**-0.77
                           * (4.05 + 2.695*(- 2.156 + 1.509/0.12
                                            - 0.198/0.12**2 + 0.011/0.12**3)))

        A_lambda[mask2] = (4.05 + 2.695*(- 2.156
                                         + 1.509/wavs_mic[mask2]
                                         - 0.198/wavs_mic[mask2]**2
                                         + 0.011/wavs_mic[mask2]**3))

        A_lambda[mask3] = 2.659*(-1.857 + 1.040/wavs_mic[mask3]) + 4.05

        return A_lambda/4.05

    def _cf00_attenuation(self, wavs):
        """ Calculate the ratio A(lambda)/A(V) for the model based on
        Charlot + Fall (2000) described in Carnall et al. (2018). This
        will be raised to some power when the transmission values are
        requested to produce a variable-slope dust curve. """

        return 5500./wavs

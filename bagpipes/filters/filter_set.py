from __future__ import print_function, division, absolute_import

import numpy as np

from .. import utils


class filter_set(object):
    """ Class for loading and manipulating sets of filter curves. This
    is where integration over filter curves to get photometry happens.

    Parameters
    ----------

    filt_list : list
        List of strings containing paths from the working directory to
        files where filter curves are stored. The filter curve files
        should contain an array of wavelengths in Angstroms followed by
        a column of relative transmission values.
    """

    def __init__(self, filt_list):
        self.filt_list = filt_list
        self.wavelengths = None
        self._load_filter_curves()
        self._calculate_min_max_wavelengths()
        self._calculate_effective_wavelengths()

    def _load_filter_curves(self):
        """ Loads filter files for the specified filt_list and truncates
        any zeros from either of their edges. """

        self.filt_dict = {}

        for filt in self.filt_list:
            try:
                self.filt_dict[filt] = np.loadtxt(filt, usecols=(0, 1))

            except IOError:
                self.filt_dict[filt] = np.loadtxt(utils.install_dir + "/"
                                                  + filt, usecols=(0, 1))

            while self.filt_dict[filt][0, 1] == 0.:
                self.filt_dict[filt] = self.filt_dict[filt][1:, :]

            while self.filt_dict[filt][-1, 1] == 0.:
                self.filt_dict[filt] = self.filt_dict[filt][:-1, :]

    def _calculate_min_max_wavelengths(self):
        """ Finds the min and max wavelength values across all of the
        filter curves. """

        self.min_phot_wav = 9.9*10**99
        self.max_phot_wav = 0.

        for filt in self.filt_list:
            min_wav = (self.filt_dict[filt][0, 0]
                       - 2*(self.filt_dict[filt][1, 0]
                       - self.filt_dict[filt][0, 0]))

            max_wav = (self.filt_dict[filt][-1, 0]
                       + 2*(self.filt_dict[filt][-1, 0]
                       - self.filt_dict[filt][-2, 0]))

            if min_wav < self.min_phot_wav:
                self.min_phot_wav = min_wav

            if max_wav > self.max_phot_wav:
                self.max_phot_wav = max_wav

    def _calculate_effective_wavelengths(self):
        """ Calculates effective wavelengths for each filter curve. """

        self.eff_wavs = np.zeros(len(self.filt_list))

        for i in range(len(self.filt_list)):
            filt = self.filt_list[i]
            dlambda = utils.make_bins(self.filt_dict[filt][:, 0])[1]
            filt_weights = dlambda*self.filt_dict[filt][:, 1]
            self.eff_wavs[i] = np.sqrt(np.sum(filt_weights)
                                       / np.sum(filt_weights
                                       / self.filt_dict[filt][:, 0]**2))

    def resample_filter_curves(self, wavelengths):
        """ Resamples the filter curves onto a new set of wavelengths
        and creates a 2D array of filter curves on this sampling. """

        self.wavelengths = wavelengths  # Wavelengths for new sampling

        # Array containing filter profiles on new wavelength sampling
        self.filt_array = np.zeros((wavelengths.shape[0], len(self.filt_list)))

        # Array containing the width in wavelength space for each point
        self.widths = utils.make_bins(wavelengths)[1]

        for i in range(len(self.filt_list)):
            filt = self.filt_list[i]
            self.filt_array[:, i] = np.interp(wavelengths,
                                              self.filt_dict[filt][:, 0],
                                              self.filt_dict[filt][:, 1],
                                              left=0, right=0)

    def get_photometry(self, spectrum, redshift, unit_conv=None):
        """ Calculates photometric fluxes. The filters are first re-
        sampled onto the same wavelength grid with transmission values
        blueshifted by (1+z). This is followed by an integration over
        the observed spectrum in the rest frame:

        flux = integrate[(f_lambda*lambda*T(lambda*(1+z))*dlambda)]
        norm = integrate[(lambda*T(lambda*(1+z))*dlambda))]
        photometry = flux/norm

        lambda:            rest-frame wavelength array
        f_lambda:          observed spectrum
        T(lambda*(1+z)):   transmission of blueshifted filters
        dlambda:           width of each wavelength bin

        The integrals over all filters are done in one array operation
        to improve the speed of the code.
        """

        if self.wavelengths is None:
            raise ValueError("Please use resample_filter_curves method to set"
                             + " wavelengths before calculating photometry.")

        redshifted_wavs = self.wavelengths*(1. + redshift)

        # Array containing blueshifted filter curves
        filters_z = np.zeros_like(self.filt_array)

        # blueshift filter curves to sample right bit of rest frame spec
        for i in range(len(self.filt_list)):
            filters_z[:, i] = np.interp(redshifted_wavs, self.wavelengths,
                                        self.filt_array[:, i],
                                        left=0, right=0)

        # Calculate numerator of expression
        flux = np.expand_dims(spectrum*self.widths*self.wavelengths, axis=1)
        flux = np.sum(flux*filters_z, axis=0)

        # Calculate denominator of expression
        norm = filters_z*np.expand_dims(self.widths*self.wavelengths, axis=1)
        norm = np.sum(norm, axis=0)

        photometry = np.squeeze(flux/norm)

        # This is a little dodgy as pointed out by Ivo, it should depend
        # on the spectral shape however only currently used for UVJ mags
        if unit_conv == "cgs_to_mujy":
            photometry /= (10**-29*2.9979*10**18/self.eff_wavs**2)

        return photometry

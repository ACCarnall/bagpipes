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
            self.eff_wavs[i] = np.sqrt(np.sum(filt_weights*self.filt_dict[filt][:, 0])
                                       / np.sum(filt_weights
                                       / self.filt_dict[filt][:, 0]))

    def resample_filter_curves(self, wavelengths):
        """ Resamples the filter curves onto a new set of wavelengths
        and creates a 2D array of filter curves on this sampling. """

        self.wavelengths = wavelengths  # Wavelengths for new sampling

        # Array containing filter profiles on new wavelength sampling
        self.filt_array = np.zeros((wavelengths.shape[0], len(self.filt_list)))

        # Array containing the width in wavelength space for each point
        self.widths = utils.make_bins(wavelengths)[1]
        self.widths_wavs = self.widths*self.wavelengths

        for i in range(len(self.filt_list)):
            filt = self.filt_list[i]
            self.filt_array[:, i] = np.interp(wavelengths,
                                              self.filt_dict[filt][:, 0],
                                              self.filt_dict[filt][:, 1],
                                              left=0, right=0)

        self._cached_redshift = None

    def _set_redshift(self, redshift):
        """ Calculate and store the quantities needed for photometry
        which depend only on redshift. T(lambda*(1+z)) is a linear
        interpolation between filt_array rows ind and ind+1 with weights
        (1-frac) and frac. Rather than building blueshifted filter curves,
        get_photometry moves the integration weights onto those rows,
        which is equivalent and handles all filters at once. """

        weights = self.widths_wavs

        if redshift == 0.:
            weights_z = weights

        else:
            redshifted_wavs = self.wavelengths*(1. + redshift)
            n_wavs = self.wavelengths.shape[0]

            ind = np.searchsorted(self.wavelengths, redshifted_wavs,
                                  side="right") - 1
            ind = np.clip(ind, 0, n_wavs - 2)

            frac = ((redshifted_wavs - self.wavelengths[ind])
                    / (self.wavelengths[ind+1] - self.wavelengths[ind]))

            # Zero transmission beyond the range of the filter curves
            inside = ((redshifted_wavs >= self.wavelengths[0])
                      & (redshifted_wavs <= self.wavelengths[-1]))

            self._ind = ind
            self._inside = inside.astype(float)
            self._high_frac = frac*self._inside

            high = weights*self._high_frac
            low = weights*self._inside - high
            weights_z = (np.bincount(ind, low, n_wavs)
                         + np.bincount(ind+1, high, n_wavs))

        # Denominator of the photometry expression
        self._norm = weights_z @ self.filt_array
        self._cached_redshift = redshift

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

        # Integration weights (lambda*dlambda) for the numerator
        weights = spectrum*self.widths_wavs

        # Everything else depends only on redshift, so is re-used if the
        # redshift does not change (e.g. when fitting at fixed redshift).
        if redshift != self._cached_redshift:
            self._set_redshift(redshift)

        # At zero redshift the filter curves need no blueshifting.
        if redshift == 0.:
            weights_z = weights

        # Otherwise move the integration weights onto the filt_array rows
        # used to interpolate the blueshifted filter curves.
        else:
            n_wavs = self.wavelengths.shape[0]
            high = weights*self._high_frac
            low = weights*self._inside - high
            weights_z = (np.bincount(self._ind, low, n_wavs)
                         + np.bincount(self._ind+1, high, n_wavs))

        # Calculate numerator and denominator of expression
        flux = weights_z @ self.filt_array
        norm = self._norm

        photometry = np.squeeze(flux/norm)

        # This is a little dodgy as pointed out by Ivo, it should depend
        # on the spectral shape however only currently used for UVJ mags
        if unit_conv == "cgs_to_mujy":
            photometry /= (10**-29*2.9979*10**18/self.eff_wavs**2)

        return photometry

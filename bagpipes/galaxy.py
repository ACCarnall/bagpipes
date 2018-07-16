from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os

from . import utils
from . import plotting


class galaxy:
    """ Load up observational data into Bagpipes.

    Parameters
    ----------

    ID : str
        string denoting the ID of the object to be loaded. This will be
        passed to load_data.

    load_data : function
        User-defined function which should take ID as an argument and
        return spectroscopic and/or photometric data. Spectroscopy
        should come first and be formatted as an array containing first
        a column of wavelengths in Angstroms, then secondly a column of
        fluxes in erg/s/cm^2/A and finally a column of flux errors
        in the same units. Photometry should come second and be
        formatted as an array containing first a column of fluxes in
        microjanskys and a column of flux errors in the same units.

    filt_list : list - optional
        A list of paths to filter curve files, which should contain a
        column of wavelengths in angstroms followed by a column of
        transmitted fraction values. Only required if photometric output
        is desired.

    spectrum_exists : bool
        If you do not have a spectrum for this object, set this to
        False. In this case, load_data should only return photometry.

    photometry_exists : bool
        If you do not have photometry for this object, set this to
        False. In this case, load_data should only return a spectrum.
    """

    def __init__(self, ID, load_data, no_of_spectra=1, out_units="ergscma",
                 spectrum_exists=True, photometry_exists=True, filt_list=None):

        self.ID = str(ID)
        self.filt_list = filt_list
        self.out_units = out_units
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.no_of_spectra = no_of_spectra

        if not spectrum_exists and not photometry_exists:
            sys.exit("Bagpipes: Object must have at least some data.")

        elif spectrum_exists and not photometry_exists:
            if self.no_of_spectra == 1:
                self.spectrum = load_data(self.ID)

            else:
                data = load_data(self.ID)
                self.spectrum = data[0]
                self.extra_spectra = []
                for i in range(len(data)-1):
                    self.extra_spectra.append(data[i+1])

        elif photometry_exists and not spectrum_exists:
            phot_nowavs = load_data(self.ID)
            self.no_of_spectra = 0

        else:
            if self.no_of_spectra == 1:
                self.spectrum, phot_nowavs = load_data(self.ID)

            else:
                data = load_data(self.ID)
                self.spectrum = data[0]
                self.extra_spectra = []
                phot_nowavs = data[-1]
                for i in range(len(data)-2):
                    self.extra_spectra.append(data[i+1])

        if photometry_exists:
            self.photometry = np.zeros(3*len(phot_nowavs))
            self.photometry.shape = (phot_nowavs.shape[0], 3)
            self.photometry[:, 1:] = phot_nowavs
            self._get_eff_wavs()

        const = 10**-29*2.9979*10**18

        if self.out_units == "ergscma" and photometry_exists:
            self.photometry[:, 1] *= (const/self.photometry[:, 0]**2)
            self.photometry[:, 2] *= (const/self.photometry[:, 0]**2)

        elif self.out_units == "mujy" and spectrum_exists:
            self.spectrum[:, 1] /= (const/self.spectrum[:, 0]**2)
            self.spectrum[:, 2] /= (const/self.spectrum[:, 0]**2)

            if self.no_of_spectra != 1:
                for i in range(len(self.extra_spectra)):
                    extra_spec = self.extra_spectra[i]
                    extra_spec[:, 1] /= (const/extra_spec[:, 0]**2)
                    extra_spec[:, 2] /= (const/extra_spec[:, 0]**2)

        # Mask the regions of the spectrum specified in [ID].mask.
        if self.spectrum_exists:
            self.spectrum = self._mask(self.spectrum)

            if self.no_of_spectra > 1:
                for i in range(len(self.extra_spectra)):
                    self.extra_spectra[i] = self._mask(self.extra_spectra[i])

            # Removes points at the edges of the spectrum with zero flux
            startn = 0
            while self.spectrum[startn, 1] == 0.:
                startn += 1

            endn = 1
            while self.spectrum[-endn-1, 1] == 0.:
                endn += 1

            self.spectrum = self.spectrum[startn:-endn, :]

    def _get_eff_wavs(self):
        """ Loads filter files from filt_list and calculates effective
        wavelength values which are added to self.photometry """
        self.eff_wavs = np.zeros(len(self.filt_list))

        for i in range(len(self.photometry)):
            filt = np.loadtxt(self.filt_list[i])
            dlambda = utils.make_bins(filt[:, 0])[1]

            num = np.sqrt(np.sum(dlambda*filt[:, 1]))
            denom = np.sqrt(np.sum(dlambda*filt[:, 1]/filt[:, 0]/filt[:, 0]))
            self.eff_wavs[i] = np.round(num/denom, 1)

        self.photometry[:, 0] = self.eff_wavs

    def _mask(self, spec):
        """ Set the error spectrum to infinity in masked regions. """

        if not os.path.exists("masks/" + self.ID + "_mask"):
            return spec

        mask = np.loadtxt("masks/" + self.ID + "_mask")
        if len(mask.shape) == 1:
            wl_mask = (spec[:, 0] > mask[0]) & (spec[:, 0] < mask[1])
            if spec[wl_mask, 2].shape[0] is not 0:
                spec[wl_mask, 2] = 9.9*10**99.

        if len(mask.shape) == 2:
            for i in range(mask.shape[0]):
                wl_mask = (spec[:, 0] > mask[i, 0]) & (spec[:, 0] < mask[i, 1])
                if spec[wl_mask, 2].shape[0] is not 0:
                    spec[wl_mask, 2] = 9.9*10**99.

        return spec

    def plot(self, show=True):
        return plotting.plot_galaxy(self, show=show)

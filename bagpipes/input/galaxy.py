from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os

from .. import plotting
from .. import filters


class galaxy:
    """ A container for observational data loaded into Bagpipes.

    Parameters
    ----------

    ID : str
        string denoting the ID of the object to be loaded. This will be
        passed to load_data.

    load_data : function
        User-defined function which should take ID as an argument and
        return spectroscopic and/or photometric data. Spectroscopy
        should come first and be an array containing a column of
        wavelengths in Angstroms, then a column of fluxes and finally a
        column of flux errors. Photometry should come second and be an
        array containing a column of fluxes and a column of flux errors.

    filt_list : list - optional
        A list of paths to filter curve files, which should contain a
        column of wavelengths in angstroms followed by a column of
        transmitted fraction values. Only needed for photometric data.

    spectrum_exists : bool - optional
        If you do not have a spectrum for this object, set this to
        False. In this case, load_data should only return photometry.

    photometry_exists : bool - optional
        If you do not have photometry for this object, set this to
        False. In this case, load_data should only return a spectrum.

    spec_units : str - optional
        Units of the input spectrum, defaults to ergs s^-1 cm^-2 A^-1,
        "ergscma". Other units (microjanskys; mujy) will be converted to
        ergscma by default within the class (see out_units).

    phot_units : str - optional
        Units of the input photometry, defaults to microjanskys, "mujy"
        The photometry will be converted to ergscma by default within
        the class (see out_units).

    out_units : str - optional
        Units to convert the inputs to within the class. Defaults to
        ergs s^-1 cm^-2 A^-1, "ergscma".
    """

    def __init__(self, ID, load_data, spec_units="ergscma", phot_units="mujy",
                 spectrum_exists=True, photometry_exists=True, filt_list=None,
                 out_units="ergscma"):

        self.ID = str(ID)
        self.phot_units = phot_units
        self.spec_units = spec_units
        self.out_units = out_units
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.filt_list = filt_list

        if not spectrum_exists and not photometry_exists:
            sys.exit("Bagpipes: Object must have at least some data.")

        elif spectrum_exists and not photometry_exists:
            self.spectrum = load_data(self.ID)
            self.spec_wavs = self.spectrum[:,0]

        elif photometry_exists and not spectrum_exists:
            phot_nowavs = load_data(self.ID)
            self.spec_wavs = None

        else:
            self.spectrum, phot_nowavs = load_data(self.ID)
            self.spec_wavs = self.spectrum[:,0]

        if photometry_exists:
            self.filter_set = filters.filter_set(filt_list)
            self.photometry = np.c_[self.filter_set.eff_wavs, phot_nowavs]

        # Perform any unit conversions.
        self._convert_units()

        # Mask the regions of the spectrum specified in masks/[ID].mask
        if self.spectrum_exists:
            self.spectrum = self._mask(self.spectrum)

            # Remove points at the edges of the spectrum with zero flux.
            startn = 0
            while self.spectrum[startn, 1] == 0.:
                startn += 1

            endn = 1
            while self.spectrum[-endn-1, 1] == 0.:
                endn += 1

            self.spectrum = self.spectrum[startn:-endn, :]

    def _convert_units(self):
        """ Convert between ergs s^-1 cm^-2 A^-1 and microjanskys if
        there is a difference between the specified input and output
        units. """

        conversion = 10**-29*2.9979*10**18/self.photometry[:, 0]**2

        if self.spectrum_exists:
            if not self.spec_units == self.out_units:
                if self.spec_units == "ergscma":
                    self.spectrum[:, 1] /= conversion
                    self.spectrum[:, 2] /= conversion

                elif spec_units == "mujy":
                    self.spectrum[:, 1] *= conversion
                    self.spectrum[:, 2] *= conversion

        if self.photometry_exists:
            if not self.phot_units == self.out_units:
                if self.phot_units == "ergscma":
                    self.spectrum[:, 1] /= conversion
                    self.spectrum[:, 2] /= conversion

                elif self.phot_units == "mujy":
                    self.photometry[:, 1] *= conversion
                    self.photometry[:, 2] *= conversion

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

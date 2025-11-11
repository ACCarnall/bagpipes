from __future__ import print_function, division, absolute_import

import numpy as np
import os

from .. import plotting
from .. import filters

from .spectral_indices import measure_index


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

    load_line_fluxes : function - optional
        Load observed line fluxes for a galaxy. The function should
        return a list of line labels in Cloudy format, as well as an
        array with a column of flux values in erg/s/cm^2/AA and a column
        of corresponding uncertainties in the same units. It is not
        recommended to use this functionality at the same time as loading
        and fitting observed spectroscopic data with the code.

    index_list : list - optional
        list of dicts containining definitions for spectral indices.

    load_indices : function or str - optional
        Load spectral index information for the galaxy. This can either
        be a function which takes the galaxy ID and returns index values
        in the same order as they are defined in index_list, or the str
        "from_spectrum", in which case the code will measure the indices
        from the observed spectrum for the galaxy.

    index_redshift : float - optional
        Observed redshift for this galaxy. This is only ever used if the
        user requests the code to calculate spectral indices from the
        observed spectrum.


    """

    def __init__(self, ID, load_data=None, spec_units="ergscma", phot_units="mujy",
                 spectrum_exists=True, photometry_exists=True, filt_list=None,
                 out_units="ergscma", load_line_fluxes=None, load_indices=None,
                 index_list=None, index_redshift=None,
                 input_spec_cov_matrix=False):

        self.ID = str(ID)
        self.phot_units = phot_units
        self.spec_units = spec_units
        self.out_units = out_units
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.filt_list = filt_list
        self.spec_wavs = None
        self.line_labels = None
        self.index_list = index_list
        self.index_redshift = index_redshift

        # Attempt to load the data from the load_data function.
        if spectrum_exists or photometry_exists:
            try:
                if not photometry_exists:
                    self.spectrum = load_data(self.ID)

                elif not spectrum_exists:
                    phot_nowavs = load_data(self.ID)

                else:
                    self.spectrum, phot_nowavs = load_data(self.ID)

            except TypeError:
                    print("load_data did not return expected outputs, did you "
                          "forget to set one or both of photometry_exists and "
                          "spectrum_exists to False?")
                    raise

        # If photometry is provided, add filter effective wavelengths to array
        if self.photometry_exists:
            self.filter_set = filters.filter_set(filt_list)
            self.photometry = np.c_[self.filter_set.eff_wavs, phot_nowavs]

        # Perform setup in the case of separate covariance matrix for spectrum
        if input_spec_cov_matrix:
            self.spec_cov = self.spectrum[1]
            self.spectrum = np.c_[self.spectrum[0],
                                  np.sqrt(np.diagonal(self.spec_cov))]

            self.spec_cov_inv = np.linalg.inv(self.spec_cov)
            # self.spec_cov_det = np.linalg.det(self.spec_cov)

        else:
            self.spec_cov = None

        # Perform any unit conversions.
        self._convert_units()

        # Mask the regions of the spectrum specified in masks/[ID].mask
        if self.spectrum_exists:
            self.spectrum = self._mask(self.spectrum)
            self.spec_wavs = self.spectrum[:, 0]

            # Remove points at the edges of the spectrum with zero flux.
            startn = 0
            while self.spectrum[startn, 1] == 0.:
                startn += 1

            endn = 0
            while self.spectrum[-endn-1, 1] == 0.:
                endn += 1

            if endn == 0:
                self.spectrum = self.spectrum[startn:, :]

            else:
                self.spectrum = self.spectrum[startn:-endn, :]

            self.spec_wavs = self.spectrum[:, 0]

        # Deal with loading any emission line fluxes
        if load_line_fluxes is not None:
            self.line_labels, self.line_fluxes = load_line_fluxes(self.ID)

        # Deal with any spectral index calculations.
        if load_indices is not None:
            self.index_names = [ind["name"] for ind in self.index_list]

            if callable(load_indices):
                self.indices = load_indices(self.ID)

            elif load_indices == "from_spectrum":
                self.indices = np.zeros((len(self.index_list), 2))
                for i in range(self.indices.shape[0]):
                    self.indices[i] = measure_index(self.index_list[i],
                                                    self.spectrum,
                                                    self.index_redshift)

    def _convert_units(self):
        """ Convert between ergs s^-1 cm^-2 A^-1 and microjanskys if
        there is a difference between the specified input and output
        units. """

        if self.spectrum_exists:
            conversion = 10**-29*2.9979*10**18/self.spectrum[:, 0]**2

            if not self.spec_units == self.out_units:
                if self.spec_units == "ergscma":
                    self.spectrum[:, 1] /= conversion
                    self.spectrum[:, 2] /= conversion

                    if self.spec_cov is not None:
                        self.spec_cov /= conversion

                elif self.spec_units == "mujy":
                    self.spectrum[:, 1] *= conversion
                    self.spectrum[:, 2] *= conversion

                    if self.spec_cov is not None:
                        self.spec_cov *= conversion

        if self.photometry_exists:
            conversion = 10**-29*2.9979*10**18/self.photometry[:, 0]**2

            if not self.phot_units == self.out_units:
                if self.phot_units == "ergscma":
                    self.photometry[:, 1] /= conversion
                    self.photometry[:, 2] /= conversion

                elif self.phot_units == "mujy":
                    self.photometry[:, 1] *= conversion
                    self.photometry[:, 2] *= conversion

    def _mask(self, spec):
        """ Set the error spectrum to infinity in masked regions. """

        if not os.path.exists("masks/" + self.ID + "_mask"):
            return spec

        if self.spec_cov is not None:
            raise ValueError("Automatic masking not supported where covariance"
                             " matrix is specified, please do this manually.")

        mask = np.loadtxt("masks/" + self.ID + "_mask")
        if len(mask.shape) == 1:
            wl_mask = (spec[:, 0] > mask[0]) & (spec[:, 0] < mask[1])
            if spec[wl_mask, 2].shape[0] != 0:
                spec[wl_mask, 2] = 9.9*10**99.

        if len(mask.shape) == 2:
            for i in range(mask.shape[0]):
                wl_mask = (spec[:, 0] > mask[i, 0]) & (spec[:, 0] < mask[i, 1])
                if spec[wl_mask, 2].shape[0] != 0:
                    spec[wl_mask, 2] = 9.9*10**99.

        return spec

    def plot(self, show=True, return_y_scale=False, y_scale_spec=None):
        return plotting.plot_galaxy(self, show=show,
                                    return_y_scale=return_y_scale,
                                    y_scale_spec=y_scale_spec)

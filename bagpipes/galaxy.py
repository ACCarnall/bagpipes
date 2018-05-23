from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os

from .utils import *
from . import plotting

class Galaxy:

    """ 
    Load up observational data into Bagpipes for plotting and fitting.

    Parameters
    ----------

    ID : str
        string denoting the ID of the object to be loaded. This will be passed to load_data.

    load_data : function
        Function which takes ID, filtlist as its two arguments and returns the model spectrum and photometry. Spectrum should come first and 
        be an array with a column of wavelengths in Angstroms, a column of fluxes in erg/s/cm^2/A and a column of flux errors in the same 
        units. Photometry should come second and be an array with a column of fluxes in microjanskys and a column of flux errors in the 
        same units.

    filtlist : str
        The name of the filtlist, must be specified if photometry is to be loaded.

    spectrum_exists : bool
        If you do not have a spectrum for this object, set this to False. In this case, load_data should only return photometry.

    photometry_exists : bool
        If you do not have photometry for this object, set this to False. In this case, load_data should only return a spectrum.

    
    """

    def __init__(self, ID, load_data, filtlist=None, spectrum_exists=True, photometry_exists=True, out_units="ergscma", no_of_spectra=1):

        self.ID = ID
        self.filtlist = filtlist
        self.out_units = out_units
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.no_of_spectra = no_of_spectra

        if spectrum_exists == False and photometry_exists == False:
            sys.exit("The object must have either spectral data or photometry!")

        elif spectrum_exists == True and photometry_exists == False:
            if self.no_of_spectra == 1:
                self.spectrum = load_data(self.ID, self.filtlist)

            else:
                data = load_data(self.ID, self.filtlist)
                self.spectrum = data[0]
                self.extra_spectra = []
                for i in range(len(data)-1):
                    self.extra_spectra.append(data[i+1])

        elif spectrum_exists == False and photometry_exists == True:
            photometry_nowavs = load_data(self.ID, self.filtlist)
            self.no_of_spectra = 0

        else:
            if self.no_of_spectra == 1:
                self.spectrum, photometry_nowavs = load_data(self.ID, self.filtlist)

            else:
                data = load_data(self.ID, self.filtlist)
                self.spectrum = data[0]
                self.extra_spectra = []
                photometry_nowavs = data[-1]
                for i in range(len(data)-2):
                    self.extra_spectra.append(data[i+1])

        if photometry_exists == True:
            self.photometry = np.zeros(3*len(photometry_nowavs))
            self.photometry.shape = (photometry_nowavs.shape[0], 3)
            self.photometry[:, 1:] = photometry_nowavs
            self._get_eff_wavs()
        
        if self.out_units == "ergscma" and photometry_exists == True:
            self.photometry[:,1] *= (10**-29)*(2.9979*10**18/self.photometry[:,0]/self.photometry[:,0])
            self.photometry[:,2] *= (10**-29)*(2.9979*10**18/self.photometry[:,0]/self.photometry[:,0])

        elif self.out_units == "mujy" and spectrum_exists == True:
            self.spectrum[:,1] /= ((10**-29)*(2.9979*10**18/self.spectrum[:,0]/self.spectrum[:,0])) 
            self.spectrum[:,2] /= ((10**-29)*(2.9979*10**18/self.spectrum[:,0]/self.spectrum[:,0]))

            if self.no_of_spectra != 1:
                for i in range(len(self.extra_spectra)):
                    self.extra_spectra[i][:,1] /= ((10**-29)*(2.9979*10**18/self.extra_spectra[i][:,0]/self.extra_spectra[i][:,0])) 
                    self.extra_spectra[i][:,2] /= ((10**-29)*(2.9979*10**18/self.extra_spectra[i][:,0]/self.extra_spectra[i][:,0])) 

        if self.spectrum_exists == True:
            # Mask the regions of the spectrum which have been specified in the [ID].mask file.
            self.spectrum = self._mask_spectrum(self.spectrum)

            if self.no_of_spectra > 1:
                for i in range(len(self.extra_spectra)):
                    self.extra_spectra[i] = self._mask_spectrum(self.extra_spectra[i])

            # Removes any regions at the start end end of the spectrum if flux values are zero.
            startn = 0
            while self.spectrum[startn,1] == 0.:
                startn += 1

            endn = 1
            while self.spectrum[-endn-1,1] == 0.:
                endn += 1

            self.spectrum = self.spectrum[startn:-endn, :]
            


    """ Loads filter files for the specified filtlist and calculates effective wavelength values which are added to self.photometry """
    def _get_eff_wavs(self):
        self.filt_names = np.loadtxt(working_dir + "/pipes/filters/" + self.filtlist + ".filtlist", dtype="str")

        self.eff_wavs = np.zeros(len(self.filt_names))

        for i in range(len(self.photometry)):
            filt = np.loadtxt(working_dir + "/pipes/filters/" + self.filt_names[i])
            dlambda = make_bins(filt[:,0])[1]
            self.eff_wavs[i] = np.round(np.sqrt(np.sum(dlambda*filt[:,1])/np.sum(dlambda*filt[:,1]/filt[:,0]/filt[:,0])), 1)

        self.photometry[:,0] = self.eff_wavs

            
    """ Set the error spectrum to infinity in masked regions. """
    def _mask_spectrum(self, spectrum):
        if not os.path.exists(working_dir + "/pipes/object_masks/" + self.ID + "_mask"): #" + self.ID + "
            return spectrum

        else:
            mask = np.loadtxt(working_dir + "/pipes/object_masks/" + self.ID + "_mask") #" + self.ID + "
            if len(mask.shape) == 1:
                if spectrum[(spectrum[:,0] > mask[0]) & (spectrum[:,0] < mask[1]), 2].shape[0] is not 0:
                    spectrum[(spectrum[:,0] > mask[0]) & (spectrum[:,0] < mask[1]), 2] = 9.9*10**99.

            if len(mask.shape) == 2:
                for i in range(mask.shape[0]):
                    if spectrum[(spectrum[:,0] > mask[i,0]) & (spectrum[:,0] < mask[i,1]), 2].shape[0] is not 0:
                        spectrum[(spectrum[:,0] > mask[i,0]) & (spectrum[:,0] < mask[i,1]), 2] = 9.9*10**99.

        return spectrum

    def plot(self, show=True):
        return plotting.plot_galaxy(self, show=show)


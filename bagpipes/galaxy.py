import numpy as np
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os

import setup
import model_galaxy

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

class Galaxy:

    """ 
    Load up observational data into Bagpipes for plotting and fitting.

    Parameters
    ----------

    ID : str
        string denoting the ID of the object to be loaded. This will be passed to data_load_function.

    data_load_function : function
        Function which takes ID, field as its two arguments and returns the model spectrum and photometry. Spectrum should come first and 
        be an array with a column of wavelengths in Angstroms, a column of fluxes in erg/s/cm^2/A and a column of flux errors in the same 
        units. Photometry should come second and be an array with a column of fluxes in microjanskys and a column of flux errors in the 
        same units.

    field : str
        The name of the field, must be specified if photometry is to be loaded.

    spectrum_exists : bool
        If you do not have a spectrum for this object, set this to False. In this case, data_load_function should only return photometry.

    photometry_exists : bool
        If you do not have photometry for this object, set this to False. In this case, data_load_function should only return a spectrum.

    
    """

    def __init__(self, ID, data_load_function, field=None, spectrum_exists=True, photometry_exists=True):

        out_units="ergscma"
        no_of_spectra=1

        self.ID = ID
        self.field = field
        self.out_units = out_units
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.no_of_spectra = no_of_spectra

        if spectrum_exists == False and photometry_exists == False:
            sys.exit("The object must have either spectral data or photometry!")

        elif spectrum_exists == True and photometry_exists == False:
            if self.no_of_spectra == 1:
                self.spectrum = data_load_function(self.ID, self.field)

            else:
                data = data_load_function(self.ID, self.field)
                self.spectrum = data[0]
                self.extra_spectra = []
                for i in range(len(data)-1):
                    self.extra_spectra.append(data[i+1])

        elif spectrum_exists == False and photometry_exists == True:
            photometry_nowavs = data_load_function(self.ID, self.field)
            self.no_of_spectra = 0

        else:
            if self.no_of_spectra == 1:
                self.spectrum, photometry_nowavs = data_load_function(self.ID, self.field)

            else:
                data = data_load_function(self.ID, self.field)
                self.spectrum = data[0]
                self.extra_spectra = []
                photometry_nowavs = data[-1]
                for i in range(len(data)-2):
                    self.extra_spectra.append(data[i+1])

        if photometry_exists == True:
            self.photometry = np.zeros(3*len(photometry_nowavs))
            self.photometry.shape = (photometry_nowavs.shape[0], 3)
            self.photometry[:, 1:] = photometry_nowavs
            self.get_eff_wavs()
        
        if self.out_units == "ergscma" and photometry_exists == True:
            self.photometry[:,1] *= (10**-29)*(2.9979*10**18/self.photometry[:,0]/self.photometry[:,0])
            self.photometry[:,2] *= (10**-29)*(2.9979*10**18/self.photometry[:,0]/self.photometry[:,0])

        elif out_units == "mujy" and spectrum_exists == True:
            self.spectrum[:,1] /= ((10**-29)*(2.9979*10**18/self.spectrum[:,0]/self.spectrum[:,0])) 
            self.spectrum[:,2] /= ((10**-29)*(2.9979*10**18/self.spectrum[:,0]/self.spectrum[:,0]))

            if self.no_of_spectra != 1:
                for i in range(len(self.extra_spectra)):
                    self.extra_spectra[i][:,1] /= ((10**-29)*(2.9979*10**18/self.extra_spectra[i][:,0]/self.extra_spectra[i][:,0])) 
                    self.extra_spectra[i][:,2] /= ((10**-29)*(2.9979*10**18/self.extra_spectra[i][:,0]/self.extra_spectra[i][:,0])) 

        if spectrum_exists == True:
            # Mask the regions of the spectrum which have been specified in the [ID].mask file.
            self.spectrum = self.mask_spectrum(self.spectrum)

            if self.no_of_spectra > 1:
                for i in range(len(self.extra_spectra)):
                    self.extra_spectra[i] = self.mask_spectrum(self.extra_spectra[i])

            # Removes any regions at the start end end of the spectrum if flux values are zero.
            startn = 0
            while self.spectrum[startn,1] == 0.:
                startn += 1

            endn = 1
            while self.spectrum[-endn-1,1] == 0.:
                endn += 1

            self.spectrum = self.spectrum[startn:-endn, :]
            


    def plot(self):

        """
        Plots the data which has been loaded.
        """

        naxes = self.no_of_spectra

        if self.photometry_exists == True:
            naxes += 1

        fig, axes = plt.subplots(naxes, figsize=(14, 4*naxes))

        if naxes == 1:
            axes = [axes]

        ax1 = axes[0]
        ax2 = axes[-1]

        ax2.set_xlabel("$\mathrm{Wavelength\ (\AA)}$", size=18)

        fig.text(0.08, 0.58, "$\mathrm{f_{\lambda}}$ $\mathrm{(erg/s/cm^2/\AA)}$", size=18, rotation=90)

        # Plot spectral data
        if self.spectrum_exists == True:
            ax1.set_xlim(self.spectrum[0,0], self.spectrum[-1,0])
            ax1.set_ylim(0., 1.05*np.max(self.spectrum[:,1]))

            ax1.plot(self.spectrum[:, 0], self.spectrum[:, 1], color="dodgerblue", zorder=1)
            ax1.fill_between(self.spectrum[:, 0], self.spectrum[:, 1] - self.spectrum[:, 2], self.spectrum[:, 1] + self.spectrum[:, 2], color="dodgerblue", zorder=1, alpha=0.75, linewidth=0)

        # Plot any extra spectra
        if self.no_of_spectra > 1:
            for i in range(self.no_of_spectra-1):
                axes[i+1].plot(self.extra_spectra[i][:, 0], self.extra_spectra[i][:, 1], color="dodgerblue", zorder=1)

        # Plot photometric data
        if self.photometry_exists == True:
            ax2.set_xscale("log")
            ax2.set_ylim(0., 1.1*np.max(self.photometry[:,1]))

            for axis in axes:
                axis.errorbar(self.photometry[:,0], self.photometry[:,1], yerr=self.photometry[:,2], lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=2, color="black")
                axis.scatter(self.photometry[:,0], self.photometry[:,1], color="blue", s=75, zorder=3, linewidth=1, facecolor="blue", edgecolor="black", label="Observed Photometry")


        # Add masked regions to plots
        if os.path.exists(setup.install_dir + "/object_masks/" + self.ID + "_mask") and self.spectrum_exists:
            mask = np.loadtxt(setup.install_dir + "/object_masks/" + self.ID + "_mask")

            for j in range(self.no_of_spectra):
                if len(mask.shape) == 1:
                    axes[j].axvspan(mask[0], mask[1], color="gray", alpha=0.8, zorder=3)

                if len(mask.shape) == 2:
                    for i in range(mask.shape[0]):
                        axes[j].axvspan(mask[i,0], mask[i,1], color="gray", alpha=0.8, zorder=3)
            
        plt.show()
        plt.close(fig)



    """ Loads filter files for the specified field and calculates effective wavelength values which are added to self.photometry """
    def get_eff_wavs(self):
        filtlist = np.loadtxt(setup.install_dir + "/filters/" + self.field + "_filtlist.txt", dtype="str")
        for i in range(len(self.photometry)):
            filt = np.loadtxt(setup.install_dir + "/filters/" + filtlist[i])
            dlambda = setup.make_bins(filt[:,0])[1]
            self.photometry[i, 0] = np.round(np.sqrt(np.sum(dlambda*filt[:,1])/np.sum(dlambda*filt[:,1]/filt[:,0]/filt[:,0])), 1)


            
    """ Set the error spectrum to infinity in masked regions. """
    def mask_spectrum(self, spectrum):
        if not os.path.exists(setup.install_dir + "/object_masks/" + self.ID + "_mask"): #" + self.ID + "
            return spectrum

        else:
            mask = np.loadtxt(setup.install_dir + "/object_masks/" + self.ID + "_mask") #" + self.ID + "
            if len(mask.shape) == 1:
                if spectrum[(spectrum[:,0] > mask[0]) & (spectrum[:,0] < mask[1]), 2].shape[0] is not 0:
                    spectrum[(spectrum[:,0] > mask[0]) & (spectrum[:,0] < mask[1]), 2] = 9.9*10**99.

            if len(mask.shape) == 2:
                for i in range(mask.shape[0]):
                    if spectrum[(spectrum[:,0] > mask[i,0]) & (spectrum[:,0] < mask[i,1]), 2].shape[0] is not 0:
                        spectrum[(spectrum[:,0] > mask[i,0]) & (spectrum[:,0] < mask[i,1]), 2] = 9.9*10**99.

        return spectrum


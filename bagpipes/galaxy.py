import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from matplotlib import gridspec

import model_manager as models 

class Galaxy:

    """ 
    Load up observational data into Bagpipes for plotting and fitting.

    Parameters
    ----------

    ID : str
        string denoting the ID of the object to be loaded. This will be passed to data_load_function.

    data_load_function : function
        Function which takes ID, filtlist as its two arguments and returns the model spectrum and photometry. Spectrum should come first and 
        be an array with a column of wavelengths in Angstroms, a column of fluxes in erg/s/cm^2/A and a column of flux errors in the same 
        units. Photometry should come second and be an array with a column of fluxes in microjanskys and a column of flux errors in the 
        same units.

    filtlist : str
        The name of the filtlist, must be specified if photometry is to be loaded.

    spectrum_exists : bool
        If you do not have a spectrum for this object, set this to False. In this case, data_load_function should only return photometry.

    photometry_exists : bool
        If you do not have photometry for this object, set this to False. In this case, data_load_function should only return a spectrum.

    
    """

    def __init__(self, ID, data_load_function, filtlist=None, spectrum_exists=True, photometry_exists=True):

        out_units="ergscma"
        no_of_spectra=1

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
                self.spectrum = data_load_function(self.ID, self.filtlist)

            else:
                data = data_load_function(self.ID, self.filtlist)
                self.spectrum = data[0]
                self.extra_spectra = []
                for i in range(len(data)-1):
                    self.extra_spectra.append(data[i+1])

        elif spectrum_exists == False and photometry_exists == True:
            photometry_nowavs = data_load_function(self.ID, self.filtlist)
            self.no_of_spectra = 0

        else:
            if self.no_of_spectra == 1:
                self.spectrum, photometry_nowavs = data_load_function(self.ID, self.filtlist)

            else:
                data = data_load_function(self.ID, self.filtlist)
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

        normalisation_factor = 10**18

        naxes = self.no_of_spectra

        if self.photometry_exists == True:
            naxes += 1

        fig, axes = plt.subplots(naxes, figsize=(14, 4*naxes))

        if naxes == 1:
            axes = [axes]

        ax1 = axes[0]
        ax2 = axes[-1]

        ax2.set_xlabel("$\mathrm{log_{10}}\\Big(\lambda / \mathrm{\AA}\\Big)$", size=18)

        if naxes == 2:
            fig.text(0.06, 0.58, "$\mathrm{f_{\lambda}}\ \mathrm{/\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$", size=18, rotation=90)

        else:
            ax1.set_ylabel("$\mathrm{f_{\lambda}}\ \mathrm{/\ 10^{-18}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$", size=18)
                
        # Plot spectral data
        if self.spectrum_exists == True:
            ax1.set_xlim(self.spectrum[0,0], self.spectrum[-1,0])
            ax1.set_ylim(0., 1.05*normalisation_factor*np.max(self.spectrum[:,1]))

            ax1.plot(self.spectrum[:, 0], normalisation_factor*self.spectrum[:, 1], color="dodgerblue", zorder=1)
            ax1.fill_between(self.spectrum[:, 0], normalisation_factor*(self.spectrum[:, 1] - self.spectrum[:, 2]), normalisation_factor*(self.spectrum[:, 1] + self.spectrum[:, 2]), color="dodgerblue", zorder=1, alpha=0.75, linewidth=0)
            #######
            #ax2.plot(np.log10(self.spectrum[:, 0]), normalisation_factor*self.spectrum[:, 1], color="dodgerblue", zorder=1)
            #ax2.fill_between(np.log10(self.spectrum[:, 0]), normalisation_factor*(self.spectrum[:, 1] - self.spectrum[:, 2]), normalisation_factor*(self.spectrum[:, 1] + self.spectrum[:, 2]), color="dodgerblue", zorder=1, alpha=0.75, linewidth=0)
            #######

        # Plot any extra spectra
        if self.no_of_spectra > 1:
            for i in range(self.no_of_spectra-1):
                axes[i+1].plot(self.extra_spectra[i][:, 0], self.extra_spectra[i][:, 1], color="dodgerblue", zorder=1)

        # Plot photometric data
        if self.photometry_exists == True:

            ######
            #ax2.set_ylim(0., 1.1*normalisation_factor*np.max([np.max(self.photometry[:,1]), np.max(self.spectrum[:,1])]))
            ax2.set_ylim(0., 1.1*normalisation_factor*np.max(self.photometry[:,1]))
            ######

            ax2.set_xlim((np.log10(self.photometry[0,0])-0.025), (np.log10(self.photometry[-1,0])+0.025))
            ax2.errorbar(np.log10(self.photometry[:,0]), normalisation_factor*self.photometry[:,1], yerr=normalisation_factor*self.photometry[:,2], lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=2, color="black")
            ax2.scatter(np.log10(self.photometry[:,0]), normalisation_factor*self.photometry[:,1], color="blue", s=75, zorder=3, linewidth=1, facecolor="blue", edgecolor="black", label="Observed Photometry")

            for i in range(naxes):
                axis = axes[i]
                axis.errorbar(self.photometry[:,0], normalisation_factor*self.photometry[:,1], yerr=normalisation_factor*self.photometry[:,2], lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=2, color="black")
                axis.scatter(self.photometry[:,0], normalisation_factor*self.photometry[:,1], color="blue", s=75, zorder=3, linewidth=1, facecolor="blue", edgecolor="black", label="Observed Photometry")


        # Add masked regions to plots
        if os.path.exists(models.working_dir + "/pipes/masks/" + self.ID + "_mask") and self.spectrum_exists:
            mask = np.loadtxt(models.working_dir + "/pipes/masks/" + self.ID + "_mask")

            for j in range(self.no_of_spectra):
                if len(mask.shape) == 1:
                    axes[j].axvspan(mask[0], mask[1], color="gray", alpha=0.8, zorder=3)

                if len(mask.shape) == 2:
                    for i in range(mask.shape[0]):
                        axes[j].axvspan(mask[i,0], mask[i,1], color="gray", alpha=0.8, zorder=3)
            
        plt.savefig("/Users/adam/using_bagpipes/JWST_targets/plots/" + self.ID + "_3dhst_spec.pdf", bbox_inches="tight")

        plt.show()
        plt.close(fig)




    """ Loads filter files for the specified filtlist and calculates effective wavelength values which are added to self.photometry """
    def get_eff_wavs(self):
        filtlist = np.loadtxt(models.working_dir + "/pipes/filters/" + self.filtlist + ".filtlist", dtype="str")
        for i in range(len(self.photometry)):
            filt = np.loadtxt(models.working_dir + "/pipes/filters/" + filtlist[i])
            dlambda = models.make_bins(filt[:,0])[1]
            self.photometry[i, 0] = np.round(np.sqrt(np.sum(dlambda*filt[:,1])/np.sum(dlambda*filt[:,1]/filt[:,0]/filt[:,0])), 1)


            
    """ Set the error spectrum to infinity in masked regions. """
    def mask_spectrum(self, spectrum):
        if not os.path.exists(models.install_dir + "/object_masks/" + self.ID + "_mask"): #" + self.ID + "
            return spectrum

        else:
            mask = np.loadtxt(models.install_dir + "/object_masks/" + self.ID + "_mask") #" + self.ID + "
            if len(mask.shape) == 1:
                if spectrum[(spectrum[:,0] > mask[0]) & (spectrum[:,0] < mask[1]), 2].shape[0] is not 0:
                    spectrum[(spectrum[:,0] > mask[0]) & (spectrum[:,0] < mask[1]), 2] = 9.9*10**99.

            if len(mask.shape) == 2:
                for i in range(mask.shape[0]):
                    if spectrum[(spectrum[:,0] > mask[i,0]) & (spectrum[:,0] < mask[i,1]), 2].shape[0] is not 0:
                        spectrum[(spectrum[:,0] > mask[i,0]) & (spectrum[:,0] < mask[i,1]), 2] = 9.9*10**99.

        return spectrum


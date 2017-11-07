# This example will take you through loading photometric data into bagpipes

# You must first specify your own data load function(s) which return one or both of the following:

# - For spectral data, a three column array of wavelength values in angstroms, fluxes, then flux errors (both in erg/s/cm^2/A)
# - For photometric data, a two column array of flux and flux error values in microjanskys, bands in the same order as in the field file

# If both are being returned, the code expects the spectrum first.

# Below is an example data load function for loading some data in the bagpipes/example_data folder

# At the bottom of the file is an example of loading and plotting the data.


import bagpipes as pipes
import numpy as np 



def load_uvista(ID, field):

    cat = np.loadtxt("example_UltraVISTA_data.cat")

    obj_data = np.squeeze(cat[cat[:,0] == float(ID), :])

    photometry = np.zeros((12, 2))

    photometry[:,0] = obj_data[1:13]
    photometry[:,1] = obj_data[13:25]

    photometry *= 10.**29 # convert to microjanskys

    return photometry





if __name__ == "__main__":
    """ Galaxy objects require an ID number which is passed to the data loading function you specify. They also require the data load function
    which must take ID, field as its two arguments and return a three column spectrum array containing wavelength in Angstroms, flux and 
    flux error in ergs/s/cm^2/A and a two column photometry array with flux and flux error in microjanskys. The exception to this comes when the
    spectrum_exists or photometry_exists keyword arguments are passed as false, in which case the function need only return one of the above. """

    # Generate the galaxy object with only photometric data
    galaxy = pipes.Galaxy("24", load_uvista, field="uvista", spectrum_exists=False)

    # plot the observed photometry
    galaxy.plot()

Loading data: the data_load_func
==============================================

In order to fit observational data with Bagpipes we first need to load the data into Bagpipes, this is done by initialising a **Galaxy** object. To do this you'll need to write a **data_load_func**, which supplies the data to Bagpipes in the correct format. 

The **data_load_func** should take an ID string (usually a number) and the name of the filtlist as arguments and return the observed spectrum and photometry. The spectrum should come first and be an array with a column of wavelengths in Angstroms, a column of fluxes in erg/s/cm^2/A and a column of flux errors in the same units. Photometry should come second and be an array with a column of fluxes in microjanskys and a column of flux errors in the same units (if you don't have a spectrum or photometry see below).

An example **data_load_func** would be 

.. code:: python

	def load_uvista(ID, filtlist):

	    cat = np.loadtxt("example_UltraVISTA_data.cat")

	    obj_data = np.squeeze(cat[cat[:,0] == float(ID), :])

	    photometry = np.zeros((12, 2))

	    photometry[:,0] = obj_data[1:13]
	    photometry[:,1] = obj_data[13:25]

	    photometry *= 10.**29 # convert to microjanskys

	    spectrum = np.loadtxt("example_spectra/spectrum_" + ID + ".txt")

	    return spectrum, photometry

You then generate an instance of the Galaxy class with:

.. code:: python

	import bagpipes as pipes

	galaxy = pipes.Galaxy("22", load_uvista, filtlist="uvista")

You can plot the data using:

.. code:: python

	galaxy.plot()

By default Bagpipes expects both a spectrum and photometry, if you don't have one of them you can set one of the keywords **spectrum_exists** or **photometry_exists** to False and the code will only expect to be passed the remaining data type by **data_load_func** e.g.:

.. code:: python

	import numpy as np 
	import bagpipes as pipes

	def load_uvista(ID, filtlist):

	    cat = np.loadtxt("example_UltraVISTA_data.cat")

	    obj_data = np.squeeze(cat[cat[:,0] == float(ID), :])

	    photometry = np.zeros((12, 2))

	    photometry[:,0] = obj_data[1:13]
	    photometry[:,1] = obj_data[13:25]

	    photometry *= 10.**29 # convert to microjanskys

	    return photometry




	galaxy = pipes.Galaxy("22", load_uvista, filtlist="uvista", spectrum_exists=False)

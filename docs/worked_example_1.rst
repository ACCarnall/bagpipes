Worked example 1: UC Riverside
==============================

This example takes you through loading and fitting a model to a catalogue of photometric data. The complete source code can be found `here <https://github.com/ACCarnall/bagpipes_examples/tree/master/UC_Riverside>`_.

The example was set up primarily for the April 2018 conference `"The Art of Measuring Physical Parameters in Galaxies" <https://sites.google.com/site/candelsworkshop/home>`_ at UC Riverside.

The photometry catalogue in this example is the first catalogue of CANDELS data posted on the conference website `here <https://sites.google.com/site/candelsworkshop/sedfitting-tools/catalogs>`__.


Loading up the catalogue
------------------------

The example ``load_data`` function written to load data from the first catalogue is in the `UCR_load_data.py <https://github.com/ACCarnall/bagpipes_examples/blob/master/UC_Riverside/UCR_load_data.py>`_ file:

.. code:: python

	import numpy as np 
	import sys

	def load_CANDELS_GDSS(ID, filtlist):

		IDlist = np.loadtxt("CANDELS_GDSS_workshop.dat", usecols=(0))

		if not np.max(IDlist == float(ID)):
			sys.exit("Object not found in catalogue")

		row = np.argmax(IDlist == float(ID))

		phot_mag, phot_mujy = np.zeros((17, 2)), np.zeros((17, 2))

		phot_mag[:,0] = np.loadtxt("CANDELS_GDSS_workshop.dat", usecols=(2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34))[row,:]
		phot_mag[:,1] = np.loadtxt("CANDELS_GDSS_workshop.dat", usecols=(3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35))[row,:]

		# Convert units to microjanskys.
		a = np.log(10.)/2.5
		phot_mujy[phot_mag[:,1] > 0.,0] = 10**((23.9-phot_mag[phot_mag[:,1] > 0.,0])/2.5)
		phot_mujy[phot_mag[:,1] > 0.,1] = a*10**(23.9/2.5)*np.exp(-a*phot_mag[phot_mag[:,1] > 0.,0])*phot_mag[phot_mag[:,1] > 0.,1]

		# Blow up errors associated with missing photometry points.
		for i in range(phot_mag.shape[0]):
			if phot_mag[i,1] < 0.:
				phot_mujy[i,0] = 0.
				phot_mujy[i,1] = 9.9*10**99.

		# Enforce a maximum SNR of 20, or 10 for IRAC Channels.
		for i in range(phot_mag.shape[0]-4):
			if phot_mujy[i,0] > 20.*phot_mujy[i,1]:
				phot_mujy[i,1] = phot_mujy[i,0]/20.

		for i in range(phot_mag.shape[0]-4, phot_mag.shape[0]):
			if phot_mujy[i,0] > 10.*phot_mujy[i,1]:
				phot_mujy[i,1] = phot_mujy[i,0]/10.

		return phot_mujy


Fitting the data
----------------

We will fit this data with a double-power-law star-formation history model. The ``fit_instructions`` dictionary is constructed, and the fitting performed in the `UCR_fit_data.py <https://github.com/ACCarnall/bagpipes_examples/blob/master/UC_Riverside/UCR_fit_data.py>`_ file:

.. code:: python

	import numpy as np 
	import bagpipes as pipes
	import time

	from UCR_load_data import load_CANDELS_GDSS

	dust = {}
	dust["type"] = "Calzetti"
	dust["Av"] = (0.0, 4.0) 
	dust["eta"] = 2.

	nebular = {}
	nebular["logU"] = -3.

	dblplaw = {}
	dblplaw["massformed"] = (0., 13.)
	dblplaw["metallicity"] = (0.01, 2.)
	dblplaw["alpha"] = (0.01,1000.)
	dblplaw["alphaprior"] = "log_10"
	dblplaw["beta"] = (0.01,1000.)
	dblplaw["betaprior"] = "log_10"
	dblplaw["tau"] = (0.1, 15.)

	fit_instructions = {}
	fit_instructions["dust"] = dust
	fit_instructions["dblplaw"] = dblplaw
	fit_instructions["nebular"] = nebular 
	fit_instructions["redshift"] = (0., 10.)
	fit_instructions["t_bc"] = 0.01

	all_IDs = np.loadtxt("CANDELS_GDSS_workshop.dat", usecols=(0), dtype="int")

	for ID in all_IDs:

		galaxy = pipes.Galaxy(str(ID), load_CANDELS_GDSS, filtlist="UCR_cat1", spectrum_exists=False)
		fit = pipes.Fit(galaxy, fit_instructions, run="first_run")
		
		time0 = time.time()
		fit.fit(verbose=False)
		print "Fitting time:", time.time() - time0
		print "Minimum reduced chi-squared:", fit.min_chisq_red

		fit.plot_fit()
		fit.plot_corner(param_names_tolog=["dblplaw:alpha", "dblplaw:beta"])


The output plots are saved in ``pipes/plots/``. The CPU time for this kind of fit is roughly 2-3 minutes.

For the first object, the output spectral plot is `here <https://github.com/ACCarnall/bagpipes_examples/blob/master/UC_Riverside/pipes/plots/first_run/449_fit.pdf>`__. The output corner plot is `here <https://github.com/ACCarnall/bagpipes_examples/blob/master/UC_Riverside/pipes/plots/first_run/449_corner.pdf>`__.








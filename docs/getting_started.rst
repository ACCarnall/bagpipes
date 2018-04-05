.. _getting_started:

Getting Started
===============

Bagpipes is structured around three core classes. The ``Model_Galaxy`` class, which allows model galaxy spectra, photometry and emission line strengths to be generated, the ``Galaxy`` class, which allows the user to input and plot observational data, and the ``Fit`` class, which allows the data within a ``Galaxy`` object to be fitted with Bagpipes models.


Your first model galaxy spectrum
--------------------------------

Using the ``Model_Galaxy`` class to generate model galaxies is described fully in the :ref:`Making model galaxies <making-model-galaxies>` section. However, for those wishing to start quickly, we can generate and plot a simple model galaxy spectrum as follows:

.. code:: python

	import numpy as np
	import bagpipes as pipes

	exponential = {}                  # A tau model star-formation history component.
	exponential["age"] = 2.5          # Time at which star formation began (Gyr).
	exponential["tau"] = 0.5          # Timescale of decrease in star-formation (Gyr).
	exponential["massformed"] = 10.   # Log_10 of total mass formed by component (M_solar)
	exponential["metallicity"] = 1.   # Stellar metallicity (old Solar units: Z_sol = 0.02)
		
	nebular = {}                      # Include nebular emission (lines and continuum).
	nebular["logU"] = -3.             # Log_10 of the ionization parameter.

	dust = {}                         # Include dust attenuation.
	dust["type"] = "Calzetti"         # Use the Calzetti et al. (2000) law
	dust["Av"] = 0.5                  # V band attenuation (magnitudes)
	dust["eta"] = 2.                  # Multiplication of Av due to birth clouds

	model_comp = {}
	model_comp["redshift"] = 1.0      # Observed redshift
	model_comp["veldisp"] = 250.      # Velocity dispersion (km/s)
	model_comp["t_bc"] = 0.01         # Lifetime of stellar birth clouds (Gyr)
	model_comp["exponential"] = exponential
	model_comp["nebular"] = nebular
	model_comp["dust"] = dust

	wavs = np.arange(5000., 15000., 10.)  # Output wavelengths (Angstroms, observed frame)

	model = pipes.Model_Galaxy(model_comp, output_specwavs=wavs)  # Make the model

	model.sfh.plot()  # Plot the star-formation history

	model.plot()  # Plot the output spectrum


This returns firstly a plot of the star-formation history for the model, and secondly a plot of the model spectrum from 5000 -- 15000 Angstroms in the observed frame. The output spectrum is stored as ``model.spectrum``, a two column numpy array with wavelengths in Angstroms and spectral fluxes in erg/s/cm^2/A by default. To learn more, see the `Making Model Galaxies <model_galaxies/model_galaxies.html>`_ section. To move on to inputting observational data, see the `Inputting Observational Data <loading_galaxies/loading_galaxies.html>`_ section.

.. _directory-structure:

Directory structure
-------------------

This section explains the directory structure Bagpipes sets up within your working directory in order to deal with inputs and outputs. Don't worry about this too much at the moment, but this section will probably be useful to refer back to later.

Bagpipes stores output (and expects certain inputs, such as filter curves) within the ``pipes/`` subdirectory of the directory from which you run the code. This directory will be generated automatically when the ``Fit`` class is initialised, though you will need to make it yourself in advance if you are working with photometric filter curves :ref:`(further info) <filter-lists>`.

The directory structure within the working directory is as follows:

	``pipes/``
		``filters/`` - This is where the user places filter curves and filter list (.filtlist) files, which tell the code which filter curves to use when generating photometry.

		``plots/`` - Where Bagpipes saves any plots the user requests.

		``pmn_chains/`` - Where the MultiNest output files are stored (note, MultiNest struggles with long file paths so it is best to place your working directory in your home folder).

		``cats/`` - Where output catalogues generated with the ``Catalogue_Fit`` class are stored (this functionality is not currently documented).

		``object_masks/`` - Where files specifying regions of input spectra to be masked are stored (this functionality is not currently documented).


The ``plots/`` and ``pmn_chains/`` folders are further subdivided if one specifies the ``run`` keyword argument when using the ``Fit`` class, allowing multiple different models to be fit to the same objects within the same directory structure.


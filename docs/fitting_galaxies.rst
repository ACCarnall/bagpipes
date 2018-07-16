.. _fitting-observational-data:

Fitting observational data
==========================

This section describes fitting observational data with Bagpipes. This involves setting up a ``fit`` object. The two main arguments passed to ``fit`` are a ``galaxy`` object (described in the :ref:`inputting observational data <inputting-observational-data>` section) and the ``fit_instructions`` dictionary, which contains instructions on the model to be fitted to the data.

Check out the `third iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%203%20-%20Fitting%20photometric%20data%20with%20a%20simple%20model.ipynb>`_ for a quick-start guide to fitting observational data and the `fourth <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%204%20-%20Fitting%20more%20complex%20models%20to%20photometry.ipynb>`_ for some more-advanced options.

API documentation for the ``fit`` class is provided :ref:`here <fit-api>`.

.. _fit-instructions:

The fit_instructions dictionary
-------------------------------

The ``fit_instructions`` dictionary is similar to the ``model_components`` dictionary described in the :ref:`making model galaxies <making-model-galaxies>` section. All of the available arguments are the same, however instead of single numerical values, ranges and priors can be defined on parameters the user wishes to fit.

For example, the simple model built :ref:`here <model-components>` could be fitted to data by specifying the following ``fit_instructions`` dictionary:

.. code:: python
	
	burst = {}
	burst["age"] = (0., 15.)           # Vary the age between 0 and 15 Gyr
	burst["metallicity"] = (0., 2.5)   # Vary stellar metallicity between 0 and 2.5 times Solar
	burst["massformed"] = (0., 13.)    # Vary the log_10 of total mass formed from 0 to 13.

	fit_instructions = {}
	fit_instructions["burst"] = burst         # Add the burst sfh component to the model
	fit_instructions["redshift"] = (0., 10.)  # Vary observed redshift between 0 and 10

Note that the code also automatically imposes a limit that no stars can form before the big bang, so the upper limit on the prior on burst["age"] will vary with observed redshift.

Combining this with the simple example from the :ref:`inputting observational data <load-data>` section:

.. code:: python

	import bagpipes as pipes

	eg_filt_list = ["list", "of", "filters"]

	def load_data(ID, filtlist):
	    # Do some stuff to load up data for the object with the correct ID number

	    return spectrum, photometry


	ID_number = "0001"

	galaxy = pipes.galaxy(ID_number, load_data, filt_list=eg_filt_list)

	fit = pipes.fit(galaxy, fit_instructions)

There is no need to vary all of the parameters in ``fit_instructions``. Parameters can still be fixed to single values just like in ``model_components``. Additionally, parameters can be set to mirror other parameters which are fixed or fitted. For example:

.. code:: python

	burst1 = {}
	burst1["age"] = 0.1                 # Fix the age to 0.1 Gyr
	burst1["metallicity"] = (0., 2.5)   # Vary stellar metallicity between 0 and 2.5 times Solar
	burst1["massformed"] = (0., 13.)    # Vary the Log_10 of total mass formed from 0 to 13

	burst2 = {}
	burst2["age"] = 1.0                          # Fix the age to 0.1 Gyr
	burst2["metallicity"] = "burst1:metallicity" # Mirror burst1:metallicity
	burst2["massformed"] = (0., 13.)  

	fit_instructions = {}
	fit_instructions["burst1"] = burst1       # Add the burst1 sfh component to the model
	fit_instructions["burst2"] = burst2       # Add the burst2 sfh component to the model
	fit_instructions["redshift"] = (0., 10.)  # Vary observed redshift between 0 and 10


Adding priors
-------------

At the moment, all of the parameters in the above example are fitted with uniform priors by default. We can add further keys to the relevant dictionaries to specify different priors. For example, if we wanted the prior on stellar metallicity to be uniform in log_10 of the parameter:

.. code:: python
	
	burst = {}
	burst["age"] = (0., 13.)               # Vary the age between 0 and 13 Gyr.
	burst["metallicity"] = (0.01, 5.)      # Vary metallicity between 0 and 5 times Solar
	burst["metallicity_prior"] = "log_10"  # Impose logarithmic prior over the specified range
	burst["massformed"] = (0., 13.)        # Vary the log_10 of total mass formed from 0 to 13.

The list of currently available priors is:

.. code:: python
	
	component = {}
	component["parameter_prior"] = "uniform"  # Uniform prior
	component["parameter_prior"] = "log_10"   # Uniform in log_10(parameter)
	component["parameter_prior"] = "log_e"    # Uniform in log_e(parameter)
	component["parameter_prior"] = "pow_10"   # Uniform in 10**parameter
	component["parameter_prior"] = "recip"    # Uniform in 1/parameter
	component["parameter_prior"] = "recipsq"  # Uniform in 1/parameter**2

	component["parameter_prior"] = "Gaussian" # Gaussian, also requires:
	component["parameter_prior_mu"] = 0.5      # Gaussian mean
	component["parameter_prior_sigma"] = 0.1   # Gaussian standard dev.

The limits specified are still applied when a Gaussian prior is used, for example:

.. code:: python

	fit_instructions["redshift"] = (0., 1.)
	fit_instructions["redshift_prior"] = "Gaussian"
	fit_instructions["redshift_prior_mu"] = 0.7
	fit_instructions["redshift_prior_sigma"] = 0.2

will result in a Gaussian prior on redshift centred on 0.7 with standard deviation 0.2 but which is always constrained to be between 0 and 1.


Running the sampler
-------------------

The Dynesty sampler can be run using the ``fit`` method of the ``fit`` class. Nested sampling is similar to MCMC but differs in a few key ways, for example no initial starting parameters are necessary. The ``fit`` object set up :ref:`above <fit-instructions>` can be run as follows:

.. code:: python

	fit.fit(n_live=400, verbose=True)

The ``verbose`` argument is ``False`` by default, but when set to ``True`` returns the progress of the sampler. The time taken scales roughly linearly with n_live. For simple models, 200 is probably enough, but the algorithm is more likely to fail.


Obtaining fitting results
-------------------------

Once fitting is complete, Bagpipes will calculate a large number of posterior quantities. These are stored in the ``fit.posterior`` dictionary, with the first axis of every array running over the equally-weighted posterior samples. The basic compliment of posterior quantities is:

.. code:: python

	list(fit.posterior)

	['lnprob',
	 'tmw',
	 'mwa',
	 'UVJ',
	 'samples',
	 'log_evidence_err',
	 'log_evidence',
	 'sfr',
	 'sfh',
	 'maximum_likelihood',
	 'min_chisq_reduced',
	 'confidence_interval',
	 'median',
	 'chosen_samples',
	 'mass',
	 'spectrum_full',
	 'photometry',
	 'min_chisq']

Additionally posterior distributions for each fit parameter are stored. More information about using the posterior dictionary is available in the `third iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%203%20-%20Fitting%20photometric%20data%20with%20a%20simple%20model.ipynb>`_.


Saved outputs
-------------

The basics of the posterior dictionary are automatically saved in your working directory on the completion of a fit as a hdf5 file under ``pipes/posterior/<ID>.h5``. 

When the same fit is run again this posterior will be loaded by default, if you want to start over you'll need to delete the saved file or change the run (see below).


Making output plots
-------------------

Bagpipes provides several kinds of diagnostic plots. By default, these are saved in the ``pipes/plots/`` folder.

These can be generated with:

.. code:: python

	fit.plot_fit()
	fit.plot_sfh()
	fit.plot_1d_posterior()
	fit.plot_corner()

You may find some of the functions available under pipes.plotting helpful when generating your own custom plots

The run keyword argument
------------------------

Often we will want to fit a series of different models to data, changing star-formation histories, parameter limits, priors etc. In order to quickly switch between different fitting runs without deleting output posteriors we can specify the ``run`` keyword argument of ``fit``. This will cause all outputs in ``pipes/posterior/`` and ``pipes/plots/`` to be saved into a further subdirectory with the name passed as ``run``, e.g. ``pipes/posterior/<run>/``.


.. _fit-api:

API documentation: fit
----------------------

.. autoclass:: bagpipes.fit
	:members:

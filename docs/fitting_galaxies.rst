.. _fitting-observational-data:

Fitting observational data
==========================

This section describes fitting observational data with Bagpipes. This involves setting up a ``Fit`` object. The two main arguments passed to ``Fit`` are a ``Galaxy`` object (described in the :ref:`inputting observational data <inputting-observational-data>` section) and the ``fit_instructions`` dictionary, which contains instructions on the model to be fitted to the data.

API documentation for the ``Fit`` class is provided :ref:`here <fit-api>`.

.. _fit-instructions:

The fit_instructions dictionary
-------------------------------

The ``fit_instructions`` dictionary is similar to the ``model_components`` dictionary described in the :ref:`making model galaxies <making-model-galaxies>` section. All of the available arguments are the same, however instead of single numerical values, ranges and priors can be defined on parameters the user wishes to fit.

For example, the simple model built :ref:`here <model-components>` could be fitted to data by specifying the following ``fit_instructions`` dictionary:

.. code:: python
	
	burst = {}
	burst["age"] = (0., 13.)           # Vary the age between 0 and 13 Gyr
	burst["metallicity"] = (0.01, 5.)  # Vary stellar metallicity between 0 and 5 times Solar
	burst["massformed"] = (0., 13.)    # Vary the Log_10 of total mass formed from 0 to 13.

	fit_instructions = {}
	fit_instructions["burst"] = burst         # Add the burst sfh component to the model
	fit_instructions["redshift"] = (0., 10.)  # Vary bserved redshift between zero and ten

Note that the age will additionally be constrained by the age of the Universe at the fitted redshift. 

Combining this with the simple example from the :ref:`inputting observational data <load-data>` section:

.. code:: python

	import bagpipes as pipes

	def load_data(ID, filtlist):
	    # Do some stuff to load up data for the object with the correct ID number

	    return spectrum, photometry


	ID_number = "0001"

	galaxy = pipes.Galaxy(ID_number, load_data)

	fit = pipes.Fit(Galaxy, fit_instructions)

There is no need to vary all of the parameters in ``fit_instructions``. Parameters can still be fixed to single values just like in ``model_components``. Additionally, parameters can be set to mirror other parameters which are fixed or fitted. For example:

.. code:: python

	burst1 = {}
	burst1["age"] = 0.1                 # Fix the age to 0.1 Gyr
	burst1["metallicity"] = (0.01, 5.)  # Vary stellar metallicity between 0 and 5 times Solar
	burst1["massformed"] = (0., 13.)    # Vary the Log_10 of total mass formed from 0 to 13

	burst2 = {}
	burst2["age"] = 1.0                          # Fix the age to 0.1 Gyr
	burst2["metallicity"] = "burst1:metallicity" # Mirror burst1:metallicity
	burst2["massformed"] = (0., 13.)  

	fit_instructions = {}
	fit_instructions["burst1"] = burst1       # Add the burst1 sfh component to the model
	fit_instructions["burst2"] = burst2       # Add the burst2 sfh component to the model
	fit_instructions["redshift"] = (0., 10.)  # Vary bserved redshift between zero and ten


Adding priors
-------------

At the moment, all of the parameters in the above example are fitted with uniform priors by default. We can add further keys to the relevant dictionaries to specify different priors. For example, if we wanted a log_10 prior on stellar metallicity:

.. code:: python
	
	burst = {}
	burst["age"] = (0., 13.)             # Vary the age between 0 and 13 Gyr.
	burst["metallicity"] = (0.01, 5.)    # Vary metallicity between 0 and 5 times Solar
	burst["metallicityprior"] = "log_10" # Impose logarithmic prior over the specified range
	burst["massformed"] = (0., 13.)      # Vary the Log_10 of total mass formed from 0 to 13.

The list of currently available priors is:

.. code:: python
	
	component = {}
	component["parameterprior"] = "uniform"  # Uniform prior
	component["parameterprior"] = "log_10"   # Uniform in log_10(parameter)
	component["parameterprior"] = "log_e"    # Uniform in log_e(parameter)
	component["parameterprior"] = "pow_10"   # Uniform in 10**parameter
	component["parameterprior"] = "1/x"      # Uniform in 1/parameter
	component["parameterprior"] = "1/x^2"    # Uniform in 1/parameter**2

	component["parameterprior"] = "Gaussian" # Gaussian, also requires:
	component["parameterpriormu"] = 0.5      # Gaussian mean
	component["parameterpriorsigma"] = 0.1   # Gaussian standard dev.

The limits specified are still applied when a Gaussian prior is used, for example:

.. code:: python

	fit_instructions["redshift"] = (0., 1.)
	fit_instructions["redshiftprior"] = "Gaussian"
	fit_instructions["redshiftpriormu"] = 0.7
	fit_instructions["redshiftpriorsigma"] = 0.2

will result in a Gaussian prior on redshift centred on 0.7 with standard deviation 0.2 but which is always constrained to be between 0 and 1.


Running the sampler
-------------------

The MultiNest sampler can be run using the ``fit`` method of the ``Fit`` class. Nested sampling is similar to MCMC but differs in a few key ways, for example no initial starting parameters are necessary. The ``Fit`` object set up :ref:`above <fit-instructions>` can be run as follows:

.. code:: python

	fit.fit(n_live=200, sampling_efficiency="parameter", verbose=True)

The default parameters of ``n_live=400`` and ``sampling_efficiency="model"`` are safest, but can be changed to those shown above to speed things up for testing. 

The ``verbose`` argument is ``False`` by default, but when set to ``True`` returns the progress of the sampler and final confidence intervals when sampling is complete.


Obtaining fitting results
-------------------------

Once fitting is complete, Bagpipes will calculate a large number of posterior quantities. Most of these are stored in the ``Fit.posterior`` dictionary. A few key parameters are stored separately:

.. code:: python

	Fit.min_chisq               # The minimum chi-squared value found
	Fit.min_chisq_red           # The minimum reduced chi-squared value found
	Fit.best_fit_params         # The parameter values corresponding to min_chisq
	Fit.posterior_median        # Posterior median values of fitting parameters
	Fit.conf_int                # One parameter 1 sigma confidence intervals
	Fit.global_log_evidence     # Log_e of the Bayesian evidence
	Fit.global_log_evidence_err # Error on fit.global_log_evidence 

The ordering of the parameters in these lists can be obtained from the ``Fit.fit_params`` list.

The ``Fit.posterior`` dictionary contains equally-weighted posterior samples for a large number of quantities. The final dimension of each entry in ``Fit.posterior`` always runs across the posterior samples with shape ``nsamples``.

For example:

.. code:: python

	post = fit.posterior

	post[Fit.fit_params[0]] # 1D array of posterior samples for the first fit parameter.

	post["sfr"] # 1D array of star-formation rate values at time of observation

	post["living_stellar_mass"]["total"] # 1D array of total living stellar mass values

	post["living_stellar_mass"][sfh_component] # living stellar mass values for each component

	post["UVJ"] # 2D array with shape (3, nsamples) of rest-frame U, V, J magnitudes.

	post["photometry"] # 2D array with posterior flux values, only if photometry fitted

	post["spectrum"] # 2D array with posterior flux values, only if spectrum fitted

	post["spectrum_full"] # 2D array with posterior flux values for whole internal model spectrum

This list is not exhaustive, extra posterior quantities can be added by the user modifying the source code, or on request in new versions of the code.

As an example of using this output, to obtain the 1 sigma upper and lower bounds and median value for the current star-formation rate we could write:

.. code:: python
	
	print np.percentile(fit.posterior["sfr"], 16)
	print np.median(fit.posterior["sfr"])
	print np.percentile(fit.posterior["sfr"], 84)



Making output plots
-------------------

Bagpipes provides two kinds of output plots, firstly spectral fit plots and secondly `corner plots <https://corner.readthedocs.io/en/latest/>`_.

These can be generated with:

.. code:: python

	fit.plot_fit()
	fit.plot_corner()

and will be saved in the ``pipes/plots/`` subdirectory of your working directory.


The run keyword argument
------------------------

Often we will want to fit a series of different models to data, changing star-formation histories, parameter limits, priors etc. In order to quickly switch between different fitting runs we can specify the ``run`` keyword argument of ``Fit``. This will cause all outputs in ``pipes/pmn_chains/`` and ``pipes/plots/`` to be saved into a further subdirectory with the name passed as ``run``.


.. _fit-api:

Fit API documentation
---------------------

.. autoclass:: bagpipes.Fit
	:members:

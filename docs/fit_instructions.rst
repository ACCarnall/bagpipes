.. _fit-instructions:

The fit_instructions dictionary
===============================

The ``fit_instructions`` dictionary is similar to the ``model_components`` dictionary described in the :ref:`making model galaxies <making-model-galaxies>` section. Available options are the same, however as well as fixed values, the user can specify parameters to be fitted by defining a prior range and probability density function.

For example, a very simple model could be fitted with the following ``fit_instructions`` dictionary:

.. code:: python

	burst = {}
	burst["age"] = (0., 15.)                  # Vary age from 0 to 15 Gyr
	burst["metallicity"] = (0., 2.5)          # Vary metallicity from 0 to 2.5 Solar
	burst["massformed"] = (0., 13.)           # Vary log_10(mass formed) from 0 to 13

	fit_instructions = {}
	fit_instructions["burst"] = burst         # Add the burst sfh component to the fit
	fit_instructions["redshift"] = (0., 10.)  # Vary observed redshift from 0 to 10

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

	burst1 = {}                                   # A burst component
	burst1["age"] = 0.1                           # Fix age to 0.1 Gyr
	burst1["metallicity"] = (0., 2.5)             # Vary metallicity from 0 to 2.5 Solar
	burst1["massformed"] = (0., 13.)              # Vary log_10(mass formed) from 0 to 13

	burst2 = {}                                   # A second burst component
	burst2["age"] = 1.0                           # Fix the age to 1.0 Gyr
	burst2["metallicity"] = "burst1:metallicity"  # Mirror burst1:metallicity
	burst2["massformed"] = (0., 13.)              # Vary log_10(mass formed) from 0 to 13

	fit_instructions = {}
	fit_instructions["burst1"] = burst1           # Add the burst1 sfh component to the fit
	fit_instructions["burst2"] = burst2           # Add the burst2 sfh component to the fit
	fit_instructions["redshift"] = (0., 10.)      # Vary observed redshift from 0 to 10


Adding priors
-------------

At the moment, all of the parameters in the above example are fitted with uniform priors by default. We can add further keys to the relevant dictionaries to specify different priors. For example, if we wanted the prior on stellar metallicity to be uniform in log_10 of the parameter:

.. code:: python

	burst = {}
	burst["age"] = (0., 13.)
	burst["metallicity"] = (0.01, 5.)
	burst["metallicity_prior"] = "log_10"
	burst["massformed"] = (0., 13.)

The list of currently available priors is:

.. code:: python

	component = {}
	component["parameter_prior"] = "uniform"   # Uniform prior
	component["parameter_prior"] = "log_10"    # Uniform in log_10(parameter)
	component["parameter_prior"] = "log_e"     # Uniform in log_e(parameter)
	component["parameter_prior"] = "pow_10"    # Uniform in 10**parameter
	component["parameter_prior"] = "recip"     # Uniform in 1/parameter
	component["parameter_prior"] = "recipsq"   # Uniform in 1/parameter**2

	component["parameter_prior"] = "Gaussian"  # Gaussian, also requires:
	component["parameter_prior_mu"] = 0.5      # Gaussian mean
	component["parameter_prior_sigma"] = 0.1   # Gaussian standard dev.

The limits specified are still applied when a Gaussian prior is used, for example:

.. code:: python

	fit_instructions["redshift"] = (0., 1.)
	fit_instructions["redshift_prior"] = "Gaussian"
	fit_instructions["redshift_prior_mu"] = 0.7
	fit_instructions["redshift_prior_sigma"] = 0.2

will result in a Gaussian prior on redshift centred on 0.7 with standard deviation 0.2 but which is always constrained to be between 0 and 1.

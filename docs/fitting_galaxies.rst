.. _fitting-observational-data:

Fitting observational data: fit
===============================

This section describes fitting observational data using the ``fit`` class. Check out the `third iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%203%20-%20Fitting%20photometric%20data%20with%20a%20simple%20model.ipynb>`_ for a quick-start guide and the `fourth <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%204%20-%20Fitting%20more%20complex%20models%20to%20photometry.ipynb>`_ for some more advanced options.

.. _fit-api:

API documentation: fit
----------------------

.. autoclass:: bagpipes.fit
	:members:

The fit_instructions dictionary
-------------------------------

The two arguments passed to the ``fit`` class are a ``galaxy`` object (described in the :ref:`loading observational data <inputting-observational-data>` section) and the ``fit_instructions`` dictionary, which contains instructions on the model to be fitted to the data.

This is very similar to the :ref:`model_components <model-components>` dictionary, however some additional options are available so that as well as being fixed, parameters can be fitted and prior probability density functions specified. A complete guide to the ``fit_instructions`` dictionary is provided :ref:`here <fit-instructions>`.

Running the sampler
-------------------

The MultiNest nested sampling algorithm can be run in order to sample from the posterior distribution using the ``fit`` method of the ``fit`` class. Nested sampling is similar to MCMC with a few key differences, for example no initial starting parameters are necessary. The code is now also compatible with the Nautilus sampling algorithm, which can be selected with the ``sampler`` keyword argument.

Resuming previous fitting runs
-------------------------------

It is worth noting that, by default, both samplers will resume fitting where possible using progress files that are automatically stored in pipes/posterior/<your run name>. This is often helpful when fitting is accidentally interrupted, but can lead to issues if you find a bug in your code and need to start fresh. To avoid this, you'll need to delete the relevant progress files.

Obtaining fitting results
-------------------------

The main output of the code is a set of samples from the posterior probability distribution for the model parameters. The code will also calculate samples for a series of derived quantites, e.g. the living stellar mass, ongoing star-formation rate etc. Samples are stored in the fit.posterior.samples dictionary. More information is available in the `third iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%203%20-%20Fitting%20photometric%20data%20with%20a%20simple%20model.ipynb>`_.


Saved outputs
-------------

The code saves basic output quantities needed to reconstruct the fit results without re-running the sampler as a hdf5 file under ``pipes/posterior/<ID>.h5``. When the same fit is run again the results of the previous sampler run will be loaded by default, and you will not be able to re-fit the data. If you want to start over you'll need to delete the saved file or change the run (see below).


Making output plots
-------------------

Bagpipes can provide several standard plots. These are saved under the ``pipes/plots/`` folder.

These can be generated with:

.. code:: python

	fit.plot_spectrum_posterior()  # Shows the input and fitted spectrum/photometry
	fit.plot_sfh_posterior()       # Shows the fitted star-formation history
	fit.plot_1d_posterior()        # Shows 1d posterior probability distributions
	fit.plot_corner()              # Shows 1d and 2d posterior probability distributions

You may find some of the functions available under pipes.plotting helpful when generating your own custom plots

The run keyword argument
------------------------

Often we will want to fit a series of different models to data, changing star-formation histories, parameter limits, priors etc. In order to quickly switch between different fitting runs without deleting output posteriors we can specify the ``run`` keyword argument of ``fit``. This will cause all outputs in ``pipes/posterior/`` and ``pipes/plots/`` to be saved into a further subdirectory with the name passed as ``run``, e.g. ``pipes/posterior/<run>/``.

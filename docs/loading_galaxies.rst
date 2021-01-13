.. _inputting-observational-data:

Loading observational data: galaxy
==================================

This section will introduce you to loading observational data. This is stored in the ``galaxy`` object. Check out the `second iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%202%20-%20Loading%20observational%20data.ipynb>`_ for a quick-start guide to loading data.

.. _galaxy-api:

API documentation: galaxy
-------------------------

.. autoclass:: bagpipes.galaxy
	:members:

.. _load-data:

The load_data function
----------------------

The most important argument passed to ``galaxy`` is the ``load_data`` function, which you will need to write to access your data files and return your data. ``load_data`` should take an ID string and return observed spectroscopic and/or photometric data in the correct format and units (see below).

For example:

.. code:: python

	import bagpipes as pipes

	eg_filt_list = ["list", "of", "filters"]

	def load_data(ID):
	    # Do some stuff to load up data for the object with the correct ID number

	    return spectrum, photometry


	ID_number = "0001"

	galaxy = pipes.galaxy(ID_number, load_data, filt_list=eg_filt_list)

	galaxy.plot()

This will plot the data returned by the ``load_data`` function.

By default, Bagpipes expects spectroscopic and photometric data to be returned by ``load_data`` in that order. If you do not have both, you must pass a keyword argument to ``galaxy``, either ``spectrum_exists=False`` or ``photometry_exists=False``.

The format of the spectrum returned by ``load_data`` should be a 2D array with three columns: wavelengths in Angstroms, fluxes in erg/s/cm^2/A and flux errors in the same units (can be changed to microJansksys with the ``spec_units`` keyword argument). These will be stored in ``galaxy.spectrum``.

The format of the photometry returned by ``load_data`` should be a 2D array with a column of fluxes in microJanskys and a column of flux errors in the same units (can be changed to erg/s/cm^2/A  with the ``phot_units`` keyword argument). The fluxes should be in the same order as the filters in your ``filt_list``. Bagpipes will calculate effective wavelengths for each filter and store these along with the input data in ``galaxy.photometry``.

For information about filter lists (``filt_list``), see the :ref:`model_galaxy <making-model-galaxies>` page.

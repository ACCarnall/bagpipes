.. _inputting-observational-data:

Inputting observational data
============================

This section will introduce you to loading observational data into Bagpipes. Observational data is stored in the ``Galaxy`` object. The most important argument passed to ``Galaxy`` is the ``load_data`` function, which you will need to write to access your data files and return observational data.

API documentation for the ``Galaxy`` class is provided :ref:`here <galaxy-api>`.

.. _load-data:

The load_data function
----------------------

The ``load_data`` function should take an ID number and filtlist (both passed as strings) and return observed spectroscopic and/or photometric data in the correct format and units.

For example:

.. code:: python

	import bagpipes as pipes

	def load_data(ID, filtlist):
	    # Do some stuff to load up data for the object with the correct ID number

	    return spectrum, photometry


	ID_number = "0001"

	galaxy = pipes.Galaxy(ID_number, load_data)

	galaxy.plot()

This will plot the data returned by the ``load_data`` function.

By default, Bagpipes expects spectroscopic and photometric data to be returned by ``load_data`` in that order. If you do not have both, you must pass a keyword argument to ``Galaxy``, either ``spectrum_exists=False`` or ``photometry_exists=False``.

The format of the spectrum returned by ``load_data`` should be a 2D array with three columns: wavelengths in Angstroms, fluxes in erg/s/cm^2/A and flux errors in the same units. These will be stored in ``Galaxy.spectrum``.

The format of the photometry returned by ``load_data`` should be a 2D array with a column of fluxes in microJanskys and a column of flux errors in the same units. If you are inputting photometry you should set up a :ref:`filter list <filter-lists>` and pass its name to ``Galaxy`` with the ``filtlist`` keyword argument. Bagpipes will calculate effective wavelengths for each filter and store these along with the input data in ``Galaxy.photometry``.

.. _galaxy-api:

Galaxy API documentation
------------------------

.. autoclass:: bagpipes.galaxy
	:members:

	
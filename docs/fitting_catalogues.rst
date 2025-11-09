Fitting multiple objects: fit_catalogue
=======================================

This section describes fitting a catalogue of objects with the same model using the ``fit_catalogue`` class. Check out the `sixth iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%206%20-%20Fitting%20catalogues%20of%20data.ipynb>`_ for a quick-start guide.

.. _catalogue-fit-api:

API documentation: fit_catalogue
--------------------------------

.. autoclass:: bagpipes.fit_catalogue
    :members:


Saving of output catalogues
---------------------------

``fit_catalogue`` will generate an output catalogue of posterior percentiles for all fit parameters plus some basic derived parameters. This is saved in the ``pipes/cats`` folder as ``<run>.fits``.


Parallelisation
---------------

Bagpipes supports MPI parallelisation using the python package mpi4py. You can run both fit or fit_catalogue using MPI, just do ``mpirun/mpiexec -n nproc python fit_with_bagpipes.py``. The default behaviour is to fit one object at a time using all available cores. This is useful for complicated models (e.g. fitting spectroscopy).

For catalogue fitting, an alternative approach is also available, in which multiple objects are fitted at once, each using one core. This option can be activated by setting the ``mpi_serial`` keyword argument of fit_catalogue to True. This is better for fitting relatively simple models to large catalogues of photometry, and can readily be scaled up to fitting catalogues of tens to hundreds of thousands of objects using ~100 cores on a computing cluster.

This feature no longer requires a special distribution of pymultinest, and will work with bagpipes >= v0.8.5 using normal distributions of pymultinest >= v2.11.

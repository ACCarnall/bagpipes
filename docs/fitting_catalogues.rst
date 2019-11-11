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

Bagpipes now supports parallelisation with MPI using the python package mpi4py. You can run both fit or fit_catalogue with MPI, just do ``mpirun/mpiexec -n nproc python fit_with_bagpipes.py``. The default behaviour is to fit one object at a time using all available cores, this is useful for complicated models (e.g. fitting spectroscopy).

For catalogue fitting an alternative approach is also available, in which multiple objects are fitted at once, each using one core. This option can be activated using the ``mpi_serial`` keyword argument of fit_catalogue. This is better for fitting relatively simple models to large catalogues of photometry. This option currently requires a slightly modified version of pymultinest, which can be downloaded from `this github repository <https://www.github.com/ACCarnall/pymultinest>`_. Please get in touch if you're having difficulty getting this to work.

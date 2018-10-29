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

``fit_catalogue`` will generate an output catalogue of posterior percentiles for all fit parameters plus some basic derived parameters. This is saved in the ``pipes/cats`` folder as ``<run>.cat``. By default the code merges results into a catalogue every tenth object. You can manually refresh the output catalogue by running the command.

.. code:: python

    import bagpipes as pipes

    run = "name_of_run"
    pipes.catalogue.merge(run)

Note that this will not interfere with a ``fit_catalogue`` instance in progress.


Parallelisation
---------------

The ``fit_catalogue`` object is designed to be run in parallel, in the sense that if multiple separate threads are running, different objects will automatically be parcelled out to different processes. There is currently no mechanism for distributing the computations for a single object amongst multiple cores, however normally the number of objects one wishes to fit will be larger than the number of available cores.


Cleaning the catalogue and killing processes
--------------------------------------------

Because of the way objects are parcelled out to different processes, if a process crashes or is killed mid-fit the merge routine will assume that this object has been successfully finished and will not attempt to fit it again.

In order to clean up any failed or killed fits the clean function will check for objects which have not been successfully completed and set them as available to be parcelled out to new processes. As part of this process, all running ``fit_catalogue`` instances will be killed. To start this process run:

.. code:: python

    import bagpipes as pipes

    run = "name_of_run"
    pipes.catalogue.clean(run)

Fitting catalogues of observational data
========================================

A situation which commonly arises is having a catalogue of objects with the same observations which we wish to fit with the same model. You could do this by using the bagpipes fit class in a for loop in order to do this yourself, but bagpipes provides a catalogue fitting interface which makes this easier, and allows for parallelisation, in the sense that different objects will automatically be parcelled out to processes running on different cores.

Check out the `sixth iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%206%20-%20Fitting%20catalogues%20of%20data.ipynb>`_ for a quick-start guide to using the catalogue_fit class.

.. _catalogue-fit-api:

API documentation: catalogue_fit
--------------------------------

.. autoclass:: bagpipes.catalogue_fit
    :members:
.. _filter-lists:

Defining filter lists
=====================

Bagpipes uses filter lists to load up filter curve files which are in turn used to generate photometry. 

To define a filter list (referred to as ``filtlist`` within the code) you'll first have to set up some of the :ref:`directory structure <directory-structure>`. Specifically, within your working directory you'll have to make a ``pipes/`` directory, and within ``pipes/`` a ``filters/`` directory.

Now, within the ``pipes/filters/`` directory you should create a file called ``<name of filter list>.filtlist``. For example, if I wanted to set up a PanSTARRS filter list, I could call my file ``PanSTARRS.filtlist``. 

In this file, you'll have to add paths from the ``pipes/filters/`` directory to the locations the filter curves you want to include are stored. In order to find the curves you want I recommend the `SVO filter profile service <http://svo2.cab.inta-csic.es/svo/theory/fps>`_.

For example, if you downloaded the PS1 grizy filters and put them in a directory called ``PanSTARRS/`` within the ``pipes/filters/`` directory, you'd need the following in ``PanSTARRS.filtlist``:

.. code::

	PanSTARRS/PS1.g
	PanSTARRS/PS1.r
	PanSTARRS/PS1.i
	PanSTARRS/PS1.z
	PanSTARRS/PS1.y

An example of this setup can be found `here <https://github.com/ACCarnall/bagpipes/tree/master/filters>`_. A filter list called UVJ has been set up, with associated filter curves within the ``UVJ/`` subfolder.

You're then all set to start generating photometry. All you need to do is specify the name of your filter list with the ``filtlist`` keyworld argument of Model_Galaxy:

.. code:: python

	model = pipes.Model_Galaxy(model_comp, filtlist="PanSTARRS")

Bagpipes will automatically load up your filter curves and generate output photometry. The model photometric fluxes in ``model.photometry`` are in the same order as the filters were specified in your ``filtlist`` file.

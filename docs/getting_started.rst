.. _getting_started:

Getting Started
===============

The best place to get started with the code is by looking at the `iPython notebook examples <https://github.com/ACCarnall/bagpipes/tree/master/examples>`_. It's a good idea to tackle them in order as the later examples build on the earlier ones.


The rest of this documentation provides a reference guide. It's a good idea to look through the pages in order, as later ones build on earlier concepts.

Bagpipes is structured around three core classes:

	- The ``model_galaxy`` class, which calculates model galaxy spectra, photometry and emission line strengths. This is described in the :ref:`making model galaxies <making-model-galaxies>` section. 

	- The ``galaxy`` class, which allows the user to input observational data. This is described in the :ref:`inputting observational data <inputting-observational-data>` section. 

	- The ``fit`` class, which allows the data within a ``galaxy`` object to be fitted with Bagpipes models. This is described in the :ref:`fitting observational data <fitting-observational-data>` section.


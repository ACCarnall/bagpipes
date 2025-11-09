.. _making-model-galaxies:

Making model spectra: model_galaxy
==================================

Model galaxy spectra and associated observables are created using the ``model_galaxy`` class. Check out the `first iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%201%20-%20Making%20model%20galaxies.ipynb>`_ for a quick-start guide to making models.

.. _model-galaxy-api:

.. autoclass:: bagpipes.model_galaxy
	:members:


The model_components dictionary
-------------------------------

The first and most important argument passed to ``model_galaxy`` is the ``model_components`` dictionary. This contains all of the physical information about the model you wish to create. A complete guide to the ``model_components`` dictionary is provided :ref:`here <model-components>`.


Getting observables - photometry
--------------------------------
In order to obtain predictions for photometric observations of a galaxy with the physical parameters defined in ``model_components`` it is necessary to define a list of filter curves through which observed fluxes should be calculated. This is done by defining a ``filt_list``.

This is simply a list of paths (absolute or from the directory in which the code is being run) to the locations at which these filter curves are stored. The filter curve files should contain wavelengths in Angstroms in their first column and relative transmission values in their second.

Let's look at a simple example of some code that creates predictions for photometry through a series of filter curves. For this to work you'd first need to put the filter curve files in the correct location. For sourcing filter curves I recommed the `SVO Filter Profile Service <http://svo2.cab.inta-csic.es/svo/theory/fps>`_.

.. code:: python

	import bagpipes as pipes
	import numpy as np

	uvista_filt_list = ["uvista/CFHT_u.txt",
                    	    "uvista/CFHT_g.txt",
                    	    "uvista/CFHT_r.txt",
                    	    "uvista/CFHT_i+i2.txt",
                    	    "uvista/CFHT_z.txt",
                    	    "uvista/subaru_z",
                    	    "uvista/VISTA_Y.txt",
                    	    "uvista/VISTA_J.txt",
                    	    "uvista/VISTA_H.txt",
                    	    "uvista/VISTA_Ks.txt",
                    	    "uvista/IRAC1",
                    	    "uvista/IRAC2"]

	model = pipes.model_galaxy(model_components, filt_list=uvista_filt_list)

	model.plot()

We now have a Bagpipes model galaxy! The final command generates a plot of the predicted fluxes.

Photometry is accessible as ``model.photometry``, which is a 1D array of flux values in erg/s/cm^2/A in the same order as the filter curves are specified in ``filt_list``. The output flux units can be converted to microJanskys using the ``model_galaxy`` keyword argument ``phot_units="mujy"``.

Getting observables - spectroscopy
----------------------------------

The process for obtaining model spectroscopy is simpler, just pass an array containing the desired wavelength sampling in Angstroms as the ``spec_wavs`` keyword argument.

.. code:: python

    import bagpipes as pipes
    import numpy as np

    obs_wavs = np.arange(2500., 7500., 5.)

    model = pipes.model_galaxy(model_components, spec_wavs=obs_wavs)

    model.plot()

The output spectrum is stored as ``model.spectrum`` which is a two column array, containing wavelengths in Angstroms and spectral fluxes in erg/s/cm^2/A by default. The output flux units can be converted to microJanskys using the ``model_galaxy`` keyword argument ``spec_units="mujy"``.

Getting observables - line fluxes
---------------------------------
Emission line fluxes are stored in the ``model_galaxy.line_fluxes`` dictionary. The list of emission features is `here <https://github.com/ACCarnall/bagpipes/blob/master/bagpipes/models/grids/cloudy_lines.txt>`_. These are only non-zero if a ``nebular`` component is added to ``model_components``.

Emission line naming conventions are the same as in Cloudy. The names in the above file are the keys for the lines in ``model_galaxy.line_fluxes``. For example, the Lyman alpha flux is under:

.. code:: python

	model.line_fluxes["H  1  1215.67A"]

Emission line fluxes are returned in units of erg/s/cm^2.


Note on units at redshift zero
------------------------------

The units specified above apply at non-zero redshift, however at redshift zero the luminosity distance is zero which would lead to a division by zero error. At redshift zero the code instead returns luminosities, in erg/s/A for spectroscopy and photometry, and erg/s for emission lines.


Updating models
---------------

Creating a new ``model_galaxy`` is relatively slow, however changing parameter values in ``model_components`` and calling the ``update`` method of ``model_galaxy`` rapidly updates the output predictions described above.

It should be noted that the ``update`` method is designed to deal with changing numerical parameter values, not with adding or removing components of the model or changing non-numerical values such as the dust attenuation type.

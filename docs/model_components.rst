.. _model-components:

The model_components dictionary
===============================

All of the physical parameters the user provides when creating a ``model_galaxy`` object are passed within the ``model_components`` dictionary. This page will take you though all of the available options. For a quick introduction take a look at the `first iPython notebook example <https://github.com/ACCarnall/bagpipes/blob/master/examples/Example%201%20-%20Making%20model%20galaxies.ipynb>`_.

In the below blocks of example code, parameters are set equal to "mandatory" if they must be specified, or equal to their default values if they do not need to be specified.

Global parameters
-----------------

There are a few global parameters which can be inserted directly into ``model_components``.


.. code:: python

	model_components = {}
	model_components["redshift"] = mandatory  # Observed redshift
	model_components["t_bc"] = 0.01           # Max age of birth clouds: Gyr
	model_components["veldisp"] = 0.          # Velocity dispersion: km/s

	model_components["sfh_comp"] = sfh_comp   # Dict containing SFH info

All other parameter values must first be placed within component dictionaries, which are then inserted into ``model_components``. Aside from observed redshift, the only other thing ``model_components`` must contain to be valid is at least one star-formation history component.

Star-formation history components
---------------------------------

Each SFH component is an individual parametric (or non-parametric) model for the SFH of the galaxy. Bagpipes can build up complex star-formation histories by superimposing multiple components. Components of the same type should be labelled sequentially in ``model_components`` e.g. ``burst1``, ``burst2`` etc.

All star-formation history components require the following mandatory keys to be specified:

.. code:: python

	sfh_comp = {}
	sfh_comp["massformed"] = mandatory   # Log_10 total stellar mass formed: M_Solar
	sfh_comp["metallicity"] = mandatory  # Metallicity: Z_sol = 0.02


All SFH components also take one or more additional parameters describing their shape. All of the available options are listed below.

.. code:: python

	burst = {}                           # Delta function burst
	burst["age"] = mandatory             # Time since burst: Gyr

	constant = {}                        # tophat function
	constant["age_max"] = mandatory      # Time since SF switched on: Gyr
	constant["age_min"] = mandatory      # Time since SF switched off: Gyr

	exponential = {}                     # Tau model e^-(t/tau)
	exponential["age"] = mandatory       # Time since SFH began: Gyr
	exponential["tau"] = mandatory       # Timescale of decrease: Gyr

	delayed = {}                         # Delayed Tau model t*e^-(t/tau)
	delayed["age"] = mandatory           # Time since SF began: Gyr
	delayed["tau"] = mandatory           # Timescale of decrease: Gyr


	lognormal = {}                       # lognormal SFH
	lognormal["tmax"] = mandatory        # Age of Universe at peak SF: Gyr
	lognormal["fwhm"] = mandatory        # Full width at half maximum SF: Gyr

	dblplaw = {}                         # double-power-law
	dblplaw["alpha"] = mandatory         # Falling slope index
	dblplaw["beta"] = mandatory          # Rising slope index
	dblplaw["tau"] = mandatory           # Age of Universe at turnover: Gyr

	iyer = {}                            # The model of Iyer et al. (2019)
	iyer["sfr"] = mandatory              # Solar masses per year
	iyer["bins"] = mandatory             # Integer
	iyer["bins_prior"] = "dirichlet"     # This prior distribution must be used
	iyer["alpha"] = mandatory            # Either integer or list of integers

	custom = {}                          # A custom array of SFR values
	custom["history"] = mandatory        # sfhist_array or "sfhist.txt": yr, M_Solar/yr

If a custom SFH component is specified, the "history" key must contain either an array or a string giving the path to a file containing the star formation history. In both cases the format is a column of ages in years followed by a column of star formation rates in Solar masses per year.

Nebular component
-----------------

The inclusion of the nebular component tells Bagpipes to include emission lines and nebular continuum emission in the model. These come from pre-computed Cloudy grids. The nebular emission model has only one free parameter, log_10 of the ionization parameter. The metallicity of the gas in the stellar birth clouds is assumed to be the same as the stars producing the ionizing flux.

.. code:: python

	nebular = {}
	nebular["logU"] = mandatory          # Log_10 of the ionization parameter.


Dust attenuation and emission component
---------------------------------------

The dust component governs attenuation and emission processes due to dust. Energy balance is assumed, such that all attenuated light is re-radiated.

Four dust attenuation models are implemented in Bagpipes, the Calzetti et al. (2000) model, the Cardelli et al. (1989) model, a model based on Charlot & Fall (2001) and the model of Salim et al. (2018). The dust emission models come from Draine + Li (2007).

.. code:: python

	dust = {}
	dust["type"] = mandatory   # Attenuation law: "Calzetti", "Cardelli", "CF00" or "Salim"
	dust["Av"] = mandatory     # Absolute attenuation in the V band: magnitudes
	dust["eta"] = 1.           # Multiplicative factor on Av for stars in birth clouds

	dust["n"] = 1.             # Power-law slope of attenuation law ("CF00" type only)

	dust["delta"] = 0.         # Deviation from Calzetti slope ("Salim" type only)
	dust["B"] = 0.             # 2175A bump strength ("Salim" type only)

	# Dust emission parameters
	dust["qpah"] = 2.          # PAH mass fraction
	dust["umin"] = 1.          # Lower limit of starlight intensity distribution
	dust["gamma"] = 0.01       # Fraction of stars at umin

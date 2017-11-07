# This example will show you how to obtain model photometry. Start in the same way as with the previous example:

import bagpipes as pipes
import numpy as np 

pipes.set_model_type("bc03_miles")

exponential = {}
exponential["age"] = 1.0 
exponential["tau"] = 0.25
exponential["mass"] = 11.0 
exponential["metallicity"] = 1.0 

dust = {}
dust["type"] = "Calzetti" 
dust["Av"] = 0.25 

model_data = {}
model_data["zred"] = 1.0 
model_data["veldisp"] = 250.
model_data["dust"] = dust
model_data["exponential"] = exponential


# Now, instead of output_specwavs we can specify a field keyword to obtain model photometry.

# Fields tell the code to look for a list of photometric filter curves in the bagpipes/filters folder called <fieldname>_filtlist.txt

# This file contains paths from this folder to filter curves for which the first column is wavelength in Angstroms and the second is transmission

# A few examples have already been set up, for example "uvista" which gives a series of filters defined in bagpipes/filters/uvista_filtlist.txt

model = pipes.Model_Galaxy(model_data, field="uvista")

#This time the plot gives us photometry points only (with the whole spectrum plotted in the background)
model.plot()


# If you would like both an output spectrum and photometry, simply specify both the field and the output_specwavs keywords

model = pipes.Model_Galaxy(model_data, field="UDS_HST", output_specwavs = np.arange(5000., 10000., 2.5))
model.plot()

# Accessing photometry and spectroscopy:

# Model photometry is stored in model.photometry, a 1D numpy array with fluxes in erg/s/A for z = 0 or erg/s/cm^2/A otherwise.

# Updating models: 

# this can be done by changing values in the model_data dictionary, then passing it to the update method of model:

model_data["exponential"]["age"] = 2.5

model.update(model_data)

# This is much faster than generating a new model object of the same type (e.g. same type of dust, sfh but with a different mass or redshift or age)

# however if the type of model is changed (e.g. new sfh component), a new model must be made.

model.plot()
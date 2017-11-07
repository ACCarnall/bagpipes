# This example takes you through fitting a model to photometric data with bagpipes.

# First we load up photometric data using the data load function specified in example_3.py and set up the galaxy object

import bagpipes as pipes
import numpy as np
from example_3 import load_uvista

galaxy = pipes.Galaxy("31", load_uvista, "uvista", spectrum_exists=False)  #load the data as in the previous example.

# Set up the fit_instructions dictionary, this is a lot like the model_data dictionary from example_1.py with a few new options
fit_instructions = {}

# This time as well as specifying fixed parameter values, we can set ranges across which parameters value should be varied and
# functional forms for the prior distribution across this range.

dust = {}
dust["type"] = "Calzetti"
dust["Av"] = (0.0, 4.0)  # A tuple tells the code to fit this parameter, the numbers are the fitting limits to be applied
dust["Avprior"] = "uniform" # The prior on the fit parameter, if not set this will automatically be uniform

nebular = {}
nebular["logU"] = -3.

exponential = {}
exponential["age"] = (0.05, 15.0)
exponential["ageprior"] = "uniform"

exponential["tau"] =  (0.3, 10.) 
exponential["tauprior"] = "uniform"

exponential["metallicity"] = (0.01, 2.5) 
exponential["metallicityprior"] = "uniform"

exponential["mass"] = (0., 15.)
exponential["massprior"] = "uniform"


# Here is an example of a second sfh component which could be added to fit_instructions:

"""
burst = {}
burst["age"] = (0., 5.)
burst["mass"] = (0., 15.)
burst["metallicity"] = "exponential:metallicity"

# String inputs are used to lock the parameter to the value of a different parameter. You must specify "component:parameter_name", e.g.
# if we had a burst component and wanted to lock the exponential age to the same value, we would write exponential["age"] = "burst1:age"
burst["metallicity"] = "exponential:metallicity" # this syntax can be used to fix the value of a parameter to another parameter

"""

fit_instructions["zred"] = (0.0, 7.0) # redshift
fit_instructions["zredprior"] = "uniform"
fit_instructions["t_bc"] = 0.01
fit_instructions["dust"] = dust
fit_instructions["exponential"] = exponential
fit_instructions["nebular"] = nebular
#fit_instructions["burst"] = burst


# For fits, one may specify a run name. 
run = "example_4"
# This will create subfolders in bagpipes/pmn_chains and in bagpipes/plots to store the multinest output and any generated plots respectively.

# Now set up the fit object, it takes first a galaxy object, then a fit_instructions dictionary
fit = pipes.Fit(galaxy, fit_instructions, run=run) # creates a fit object given the Galaxy object and instructions on the fit type

# Fit the specified model to the data with Pymultinest. One may optionally add a simplex minimisation step after this to find the max likelihood
fit.fit(simplex=True) 

# Make plots which will be saved under the "example" subfolder in the plots folder in the main bagpipes folder
fit.plot_fit()
fit.plot_corner()

# Look in the bagpipes/plots/example folder for the output plots!

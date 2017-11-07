# This example will take you through building a model galaxy with BAGPIPES and obtaining the model spectrum

import bagpipes as pipes
import numpy as np 

# Set the library of SSP models you want BAGPIPES to use, bc03_miles is the only one currently implemented, but others would be easy to add, so just ask.
pipes.set_model_type("bc03_miles")


# model_data: a dictionary containing all the information about the model you want to create.
model_data = {}

model_data["zred"] = 1.0 # The redshift of the galaxy, if z = 0, outputs are in erg/s/A, otherwise they are in erg/s/cm^2/A
model_data["veldisp"] = 250. # The velocity dispersion in km/s, this can remain unset if one is not trying to generate or fit spectral data.
model_data["t_bc"] = 0.01 # lifetime of the HII regions producing nebular emission in Gyr

# We'll now add a star formation history component, in this case an exponentially decreasing SFH:

# exponential: a dictionary containing information about a star formation history component, in this case a tau model.
exponential = {}

exponential["age"] = 1.0 # The age of the exponential component in Gyr
exponential["tau"] = 0.25 # The age of the exponential component in Gyr
exponential["metallicity"] = 1.0 # The metallicity of this component in Solar units
exponential["mass"] = 11.0 # The total mass in stars formed by this component over its history in log_10(M/M_Solar)

# Add the exponential sfh component to the model_data dictionary
model_data["exponential"] = exponential


# We'll now add dust to the model:

# dust: a dictionary containing information about the dust model to be applied to the spectrum
dust = {}

dust["type"] = "Calzetti" # Dust type, can also specify Charlot & Fall (2000) as "CF00" or Cardelli et al. (1989) as "Cardelli"
dust["Av"] = 0.25 # magnitudes of V band extinction

# Add the dust component to the model_data dictionary
model_data["dust"] = dust

# And finally nebular emission:

nebular = {}

nebular["logU"] = -3. # log_10 of the ionization parameter

# Add the nebular component to the model_data dictionary
model_data["nebular"] = nebular

# specwavs: the wavelength values you would like the output model spectrum to be sampled onto
specwavs = np.arange(5000., 15000., 2.5)

# Generate the model, passing model_data and specwavs as the keyword argoument output_specwavs
model = pipes.Model_Galaxy(model_data, output_specwavs=specwavs)

# Plot the model spectrum
model.plot()

# The output spectrum is stored as model.spectrum, it is a two column array with wavelengths in Angstroms then fluxes in erg/s/cm^2/A



# This example will take you through building a model galaxy with BAGPIPES, this is the clean (uncommented) version to show 
# how quickly this can be done, example_1.2.py is fully commented explaing what all of this code does.

import bagpipes as pipes
import numpy as np

model_data = {}
model_data["zred"] = 1.0 
model_data["veldisp"] = 250. 
model_data["t_bc"] = 0.01

exponential = {}
exponential["age"] = 1.0 
exponential["tau"] = 0.25 
exponential["metallicity"] = 1.0 #
exponential["mass"] = 11.0 

dust = {}
dust["type"] = "Calzetti"
dust["Av"] = 0.25 

nebular = {}
nebular["logU"] = -3.

model_data["exponential"] = exponential
model_data["dust"] = dust
model_data["nebular"] = nebular

specwavs = np.arange(5000., 15000., 2.5)

model = pipes.Model_Galaxy(model_data, output_specwavs=specwavs)

model.plot()
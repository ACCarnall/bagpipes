Manipulating models: the update method
======================================

When a new **Model_Galaxy** instance is generated, all of the relevant grids of models are loaded up by the code. This takes a long time as the grids are large. The fastest way to generate lots of models with the same overall structure but different values of model parameters is therefore to use the **update** method. Using the example from the `first page <model_components.html>`_ in this section we can update the model by:

.. code:: python

	import numpy as np
	import bagpipes as pipes

	dust = {}
	dust["type"] = "Calzetti"
	dust["Av"] = 0.25
	dust["eta"] = 2.0

	nebular = {}
	nebular["logU"] = -3.0

	dblplaw = {}
	dblplaw["alpha"] = 10.
	dblplaw["beta"] = 0.5
	dblplaw["tau"] = 7.0
	dblplaw["massformed"] = 11.
	dblplaw["metallicity"] = 1.0

	burst1 = {}
	burst1["age"] = 5.0
	burst1["massformed"] = 10.
	burst1["metallicity"] = 0.2

	burst2 = {}
	burst2["age"] = 1.0
	burst2["massformed"] = 9.5
	burst2["metallicity"] = 0.5

	model_comp = {}
	model_comp["redshift"] = 0.5
	model_comp["veldisp"] = 300.
	model_comp["t_bc"] = 0.01
	model_comp["nebular"] = nebular
	model_comp["dust"] = dust
	model_comp["dblplaw"] = dblplaw
	model_comp["burst1"] = burst1
	model_comp["burst2"] = burst2

	model = pipes.Model_Galaxy(model_comp, output_specwavs=np.arange(5000., 11000., 5.))

	model.plot()

	model_comp["dust"]["Av"] = 0.1
	model_comp["dblplaw"]["tau"] = 4.0
	model_comp["nebular"]["logU"] = -3.5

	model.update(model_comp)

	model.plot()

The **update** method executes in milliseconds, compared to creating the **Model_Galaxy** instance, which takes several seconds. 


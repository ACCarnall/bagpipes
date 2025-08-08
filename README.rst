**Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation**

Bagpipes is a state of the art code for generating realistic model galaxy spectra and fitting these to spectroscopic and photometric observations. For further information please see the Bagpipes documentation at `bagpipes.readthedocs.io <http://bagpipes.readthedocs.io>`_.

**Installation**

Bagpipes can be installed with pip:

.. code::

    pip install bagpipes

Please note you cannot run the code just by cloning the repository as the large grids of models aren't included.

**Sampling algorithms within Bagpipes**

The default sampler (and historically the only option) for fitting models to data is the `MultiNest <https://github.com/JohannesBuchner/MultiNest>`_ code, however this requires separate installation, which can be challenging on some systems. Bagpipes is now also compatible with the pure Python `nautilus <https://github.com/johannesulf/nautilus>`_ nested sampling algorithm, which is now installed by default along with Bagpipes, and will be automatically used for fitting if MultiNest is not installed. Even if you are used to using Bagpipes with Multinest, you may wish to try out Nautlus, as this may yield faster and/or more accurate results in some circumstances. For more information please see the `bagpipes documentation <http://bagpipes.readthedocs.io>`_.

**Published papers and citing the code**

Bagpipes is described primarily in Section 3 of `Carnall et al. (2018) <https://arxiv.org/abs/1712.04452>`_, with further development specific to spectroscopic fitting described in Section 4 of `Carnall et al. (2019b) <https://arxiv.org/abs/1903.11082>`_. These papers are the best place to start if you want to understand how the code works.

If you make use of Bagpipes, please include a citation to `Carnall et al. (2018) <https://arxiv.org/abs/1712.04452>`_ in any publications. You may also consider citing `Carnall et al. (2019b) <https://arxiv.org/abs/1903.11082>`_, particularly if you are fitting spectroscopy.

Please note development of the code has been ongoing since these works were published, so certain parts of the code are no longer as described. Please inquire if in doubt.


.. image:: docs/images/sfh_from_spec.png

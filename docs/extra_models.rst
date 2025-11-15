.. _extra-models:

Additional stellar and nebular models
===========

People may wish to run bagpipes with alternative model sets to the default BC03-2016 MILES option. It should be noted that many sets of stellar models do not cover the necessary age/metallicity/wavelength parameter space to be used effectively within bagpipes, however there are several other viable options. The most commonly requested are the `BPASS models <https://warwick.ac.uk/fac/sci/physics/research/astro/research/catalogues/bpass/>`_, which do span the necessary parameter space.

A pre-assembled stellar grid in Bagpipes format for one version of the BPASS v2.2.1 models is available from `this Google drive folder <https://drive.google.com/drive/folders/1R8SyXfqyHh4691WwAe_TF2FvYntI1PvN?usp=drive_link>`_. Nebular grids have also been pre-computed with Cloudy for these stellar models, and are also available at the same link. Note that these are Bagpipes-specific nebular grids, not those distributed by the BPASS team. A few other stellar/nebular grid versions may be placed in this folder from time to time, though these are likely to be far less well tested than the default BC03 and additional BPASS models.

It is possible to insert one's own stellar and nebular models into the code. To do this, lines 60-80 in the `config.py file <https://github.com/ACCarnall/bagpipes/blob/master/bagpipes/config.py>`_ in the bagpipes home directory need to be edited, redirecting the code to the new models. Comments in that file should assist you. Once the new stellar models are in place, new nebular grids can be computed using these as follows

.. code::

    import bagpipes as pipes
    pipes.models.making.make_cloudy_models.run_cloudy_grid()

This requires a working installation of the Cloudy code, version 23 or 25. The run_cloudy_grid function is compatible with mpi so can be run across multiple cores to speed things up.

When finished the grids will be placed in a directory called cloudy_temp_files/grids, within the directory from which the above code was run. The Bagpipes config.py file will again then need directing to these new nebular grids.
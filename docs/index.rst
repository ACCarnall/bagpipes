Bagpipes
========

Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation is a state of the art Python code for modelling galaxy spectra and fitting spectroscopic and photometric observations.

I hope you will find everything you need to know here, however feel free to get in touch if you have further questions, or to get an opinion on specific use cases.


What can Bagpipes do?
---------------------

.. image:: images/sfh_from_spec.png

Star-formation history recovery from spectroscopy (see `Carnall et al. 2019b <https://arxiv.org/abs/1903.11082>`_)

.. image:: images/z3_passive.png

Identification of z > 3 quiescent galaxies from photometry (see `Carnall et al. 2020 <https://arxiv.org/abs/2001.11975>`_)

See also `Carnall et al. (2019a) <https://arxiv.org/abs/1811.03635>`_, `Williams et al. (2019) <https://arxiv.org/abs/1905.11996>`_ and `Wild et al. (2020) <https://arxiv.org/abs/2001.09154>`_.


Source and installation
-----------------------

Bagpipes is `developed at GitHub <https://github.com/ACCarnall/bagpipes>`_, however the code cannot be installed from there, as the large model grid files aren't included in the repository. The code should instead be installed with pip:

.. code::

    pip install bagpipes


All of the code's Python dependencies will be automatically installed. The only non-Python dependency is the MultiNest nested sampling algorithm (used only for fitting). To install MultiNest see point 1 of the "on your own computer" section of the `PyMultiNest installation instructions <http://johannesbuchner.github.io/pymultinest-tutorial/install.html>`_.

In my experience, the sequence of commands necessary to install MultiNest on a mac is as follows:

.. code::

    git clone https://github.com/JohannesBuchner/MultiNest
    brew install gcc49
    export DYLD_LIBRARY_PATH="/usr/local/bin/gcc-4.9:$DYLD_LIBRARY_PATH"
    cd MultiNest/build
    cmake ..
    make
    sudo make install
    cd ../..
    rm -r MultiNest


Citation
--------

Bagpipes is described in Section 3 of `Carnall et al. (2018) <https://arxiv.org/abs/1712.04452>`_, if you make use of Bagpipes, please include a citation to this work in any publications. Please note development of the code has been ongoing since this work was published, so certain parts of the code are no longer as described.


Getting started
---------------

The best place to get started is by looking at the `iPython notebook examples <https://github.com/ACCarnall/bagpipes/tree/master/examples>`_. It's a good idea to tackle them in order as the later examples build on the earlier ones. These documentation pages contain a more complete reference guide.

Bagpipes is structured around three core classes:

 - :ref:`model_galaxy <making-model-galaxies>`: for generating model galaxy spectra
 - :ref:`galaxy <inputting-observational-data>`: for loading observational data into Bagpipes
 - :ref:`fit <fitting-observational-data>`: for fitting models to observational data.


Acknowledgements
----------------

A few of the excellent projects Bagpipes relies on are:

 - The `Bruzual \& Charlot (2003) <https://arxiv.org/abs/astro-ph/0309134>`_ stellar population models.
 - The `Draine \& Li (2007) <https://arxiv.org/abs/astro-ph/0608003>`_ dust emission models.
 - The `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest>`_ nested sampling algorithm `(Feroz et al. 2013) <https://arxiv.org/abs/1306.2144>`_
 - The `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest>`_ Python interface for Multinest `(Buchner et al. 2014) <https://arxiv.org/abs/1402.0004>`_.
 - The `Cloudy <https://www.nublado.org>`_ photoionization code `(Ferland et al. 2017) <https://arxiv.org/abs/1705.10877>`_.
 - The `Deepdish <http://deepdish.readthedocs.io>`_ HDF5 loading/saving interface.


 .. toctree::
    :maxdepth: 1
    :hidden:

    index.rst
    model_galaxies.rst
    model_components.rst
    loading_galaxies.rst
    fitting_galaxies.rst
    fit_instructions.rst
    fitting_catalogues.rst

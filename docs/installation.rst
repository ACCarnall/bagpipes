Installation
============

Bagpipes can be installed using pip. The model grids the code requires mean the distribution is too large to be hosted by PyPI, so pip must be linked through this page to the relevant source files:

.. code::

	pip install --allow-external bagpipes -f http://bagpipes.readthedocs.io/en/latest/installation.html bagpipes

This will automatically install the python package dependencies. Alternatively, the packaged source code can be downloaded from this link: `bagpipes-0.1.0 <http://dl.dropboxusercontent.com/s/i3rwy4sqb9do5xt/bagpipes-0.1.0.tar.gz?dl=0>`_.

The only remaining dependency (used only for fitting, not for generating models) is the MultiNest algorithm. To install MultiNest, clone the GitHub repository `github.com/JohannesBuchner/MultiNest <https://github.com/JohannesBuchner/MultiNest>`_, and follow the instructions in the readme file. If you run into trouble, further advice can be found `here <http://johannesbuchner.github.io/pymultinest-tutorial/install.html#on-your-own-computer>`_.


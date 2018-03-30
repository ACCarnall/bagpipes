Installation
====================================

Bagpipes setup is a fairly quick process:

1. Clone the Bagpipes `GitHub repository <https://github.com/ACCarnall/bagpipes>`_.

2. Download the model files (currently only BC03 are pre-packaged) from Google drive `here <https://drive.google.com/file/d/18Ark6Ya5URuJ2rdTsYlUieJlOz9CM0E2/view>`_, one you untar them you should have a "bc03_miles" folder. This needs to be placed in a folder called "models" which you'll have to create in the top level "bagpipes" folder.

3. Add the top level "bagpipes" folder to your **PYTHONPATH** variable.

4. Install the Python package dependencies (astropy, corner) and you're ready to run the first example file.

5. For fitting to be supported you must also install `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest>`_ and the Python interface `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest>`_ which can be installed with pip.


   
Installation
====================================

Bagpipes setup is a fairly quick process:

1. Clone the Bagpipes `GitHub repository <https://github.com/ACCarnall/bagpipes>`_.


2. Download the model files (currently only BC03 are pre-packaged) from Google drive `here <https://drive.google.com/file/d/18Ark6Ya5URuJ2rdTsYlUieJlOz9CM0E2/view>`_, untar them and put the resulting "bc03_miles" folder in the "bagpipes/models" folder.

3. Add the "bagpipes" folder to your **PYTHONPATH** variable

4. Install the Python package dependencies (astropy, corner) and you're ready to run the first example file.

5. For fitting to be supported you must also install `MultiNest <https://ccpforge.cse.rl.ac.uk/gf/project/multinest>`_ and the Python interface `PyMultiNest <https://johannesbuchner.github.io/PyMultiNest>`_ which can be installed with pip.


   
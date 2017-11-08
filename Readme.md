# BAGPIPES

### Bayesian Analysis of Galaxies for Physical Inference and Parameter EStimation

Bagpipes is a state of the art model galaxy generation framework and fitting tool. It is designed to generate a highly diverse range of complex model galaxy spectra, and to fit these models to arbitrary combinations of spectroscopic and photometric data using the MultiNest algorithm.

![](examples/example_pipes_model.jpg)

### Installation

Bagpipes setup is a fairly quick process:

1. Clone this repository.

2. Download the model files (currently only BC03 are pre-packaged) from Google drive [here](https://drive.google.com/open?id=18Ark6Ya5URuJ2rdTsYlUieJlOz9CM0E2), untar them and put the bc03_miles folder in the bagpipes/models folder.

3. Add the bagpipes folder to your PYTHONPATH variable 

4. Install the Python package dependencies (astropy, corner) and you're ready to run the first example file. 

5. For fitting to be supported you must also install [MultiNest](https://github.com/JohannesBuchner/MultiNest) and the Python interface PyMultiNest which can be installed with pip.

### Usage

The code is not currently properly documented, for which I apologise and intend to correct as soon as possible. In the mean time however a series of example files are provided in the `examples` directory from which it is fairly simple to understand the usage of the code. The first two examples take you through making models, the third through loading data and the fourth through fitting models to data.

If you have any problems please contact me at adamc@roe.ac.uk.

![](examples/example_spectral_plot.jpg)
![](examples/example_corner_plot.jpg)

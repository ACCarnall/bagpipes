from __future__ import print_function, division, absolute_import



import os
try:
    use_bpass = bool(int(os.environ['use_bpass']))
    print('use_bpass: ',bool(int(os.environ['use_bpass'])))
except KeyError:
    use_bpass = False

if use_bpass:
    print('Setup to use BPASS')
    from . import config_bpass as config
else:
    print('Setup to use BC03')
    from . import config

from . import utils

from . import models
from . import fitting
from . import filters
from . import plotting
from . import input
from . import catalogue
from . import moons

from .models.model_galaxy import model_galaxy
from .input.galaxy import galaxy
from .fitting.fit import fit

from .catalogue.fit_catalogue import fit_catalogue
from .catalogue.fit_catalogue_old import fit_catalogue_old


from .plotting import plot_corner, plot_calibration, plot_1d_posterior, plot_spectrum_posterior, plot_sfh_posterior, add_spectrum, plot_sfh, plot_csfh_posterior, plot_galaxy, general, add_sfh_posterior, add_csfh_posterior
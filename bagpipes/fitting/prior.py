from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.special import erf, erfinv


class prior(object):
    """ A class which allows for samples to be drawn from a joint prior
    distribution in several parameters and for transformations from the
    unit cube to the prior volume.

    Parameters
    ----------

    limits : list of tuples
        List of tuples containing lower and upper limits for the priors
        on each parameter.

    pdfs : list
        List of prior probability density functions which the parameters
        should be drawn from between the above limits.

    hyper_params : list of dicts
        Dictionaries containing fixed values for any hyper-parameters of
        the above prior distributions.
    """

    def __init__(self, limits, pdfs, hyper_params):
        self.limits = limits
        self.pdfs = pdfs
        self.hyper_params = hyper_params
        self.ndim = len(limits)

    def sample(self):
        """ Sample from the prior distribution. """

        cube = np.random.rand(self.ndim)

        return self.transform(cube)

    def transform(self, cube, ndim=0, nparam=0):
        """ Transform numbers on the unit cube to the prior volume. """

        # Call the relevant prior functions to draw random values.
        for i in range(self.ndim):
            prior_function = getattr(self, self.pdfs[i])
            cube[i] = prior_function(cube[i], self.limits[i],
                                     self.hyper_params[i])

        return cube

    def uniform(self, value, limits, hyper_params):
        """ Uniform prior in x where x is the parameter. """

        value = limits[0] + (limits[1] - limits[0])*value
        return value

    def log_10(self, value, limits, hyper_params):
        """ Uniform prior in log_10(x) where x is the parameter. """
        value = 10**((np.log10(limits[1]/limits[0]))*value
                     + np.log10(limits[0]))
        return value

    def log_e(self, value, limits, hyper_params):
        """ Uniform prior in log_e(x) where x is the parameter. """
        value = np.exp((np.log(limits[1]/limits[0]))*value + np.log(limits[0]))
        return value

    def pow_10(self, value, limits, hyper_params):
        """ Uniform prior in 10**x where x is the parameter. """
        value = np.log10((10**limits[1] - 10**limits[0])*value + 10**limits[0])
        return value

    def recip(self, value, limits, hyper_params):
        value = 1./((1./limits[1] - 1./limits[0])*value + 1./limits[0])
        return value

    def recipsq(self, value, limits, hyper_params):
        """ Uniform prior in 1/x**2 where x is the parameter. """
        value = 1./np.sqrt((1./limits[1]**2 - 1./limits[0]**2)*value
                           + 1./limits[0]**2)
        return value

    def Gaussian(self, value, limits, hyper_params):
        """ Gaussian prior between limits with specified mu and sigma. """
        mu = hyper_params["mu"]
        sigma = hyper_params["sigma"]

        uniform_max = erf((limits[1] - mu)/np.sqrt(2)/sigma)
        uniform_min = erf((limits[0] - mu)/np.sqrt(2)/sigma)
        value = (uniform_max-uniform_min)*value + uniform_min
        value = sigma*np.sqrt(2)*erfinv(value) + mu

        return value

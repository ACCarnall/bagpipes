from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.special import erf, erfinv
from scipy.stats import beta, t, expon


def dirichlet(r, alpha):
    """ This function samples from a Dirichlet distribution based on N-1
    independent random variables (r) in the range (0, 1). The method is
    that of http://www.arxiv.org/abs/1010.3436 by Michael Betancourt."""

    n = r.shape[0]+1
    x = np.zeros(n)
    z = np.zeros(n-1)
    alpha_tilda = np.zeros(n-1)

    if isinstance(alpha, (float, int)):
        alpha = np.repeat(alpha, n)

    for i in range(n-1):
        alpha_tilda[i] = np.sum(alpha[i+1:])

        z[i] = beta.ppf(r[i], alpha_tilda[i], alpha[i])

    for i in range(n-1):
        x[i] = np.prod(z[:i])*(1-z[i])

    x[-1] = np.prod(z)

    return np.cumsum(x)


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

    def exponential(self, value, limits, hyper_params):
        """ Exponential prior in x where x is the parameter. """

        value = expon.ppf(value, scale=hyper_params["scale"])
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

    def student_t(self, value, limits, hyper_params):

        if "df" in list(hyper_params):
            df = hyper_params["df"]
        else:
            df = 2.0

        if "loc" in list(hyper_params):
            loc = hyper_params["loc"]
        else:
            loc = 0.0

        if "scale" in list(hyper_params):
            scale = hyper_params["scale"]
        else:
            scale = 0.3

        uniform_min = t.cdf(limits[0], df=df, loc=loc, scale=scale)
        uniform_max = t.cdf(limits[1], df=df, loc=loc, scale=scale)

        value = (uniform_max-uniform_min)*value + uniform_min

        value = t.ppf(value, df=df, loc=loc, scale=scale)

        return value

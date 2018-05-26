import numpy as np

from scipy.special import erf, erfinv


def uniform(value, limits, hyper_params):
    """ Uniform prior in x where x is the parameter. """
    value = limits[0] + (limits[1] - limits[0])*value
    return value


def log_10(value, limits, hyper_params):
    """ Uniform prior in log_10(x) where x is the parameter. """
    value = 10**((np.log10(limits[1]/limits[0]))*value + np.log10(limits[0]))
    return value


def log_e(value, limits, hyper_params):
    """ Uniform prior in log_e(x) where x is the parameter. """
    value = np.exp((np.log(limits[1]/limits[0]))*value + np.log(limits[0]))
    return value


def pow_10(value, limits, hyper_params):
    """ Uniform prior in 10**x where x is the parameter. """
    value = np.log10((10**limits[1] - 10**limits[0])*value + 10**limits[0])
    return value


def recip(value, limits, hyper_params):
    value = 1./((1./limits[1] - 1./limits[0])*value + 1./limits[0])
    return value


def recipsq(value, limits, hyper_params):
    """ Uniform prior in 1/x**2 where x is the parameter. """
    value = 1./np.sqrt((1./limits[1]**2 - 1./limits[0]**2)*value
                       + 1./limits[0]**2)
    return value


def Gaussian(value, limits, hyper_params):
    """ Gaussian prior between limits with specified mu and sigma. """
    mu = hyper_params["mu"]
    sigma = hyper_params["sigma"]

    uniform_max = erf((limits[1] - mu)/np.sqrt(2)/sigma)
    uniform_min = erf((limits[0] - mu)/np.sqrt(2)/sigma)
    value = (uniform_max-uniform_min)*value + uniform_min
    value = sigma*np.sqrt(2)*erfinv(value) + mu

    return value

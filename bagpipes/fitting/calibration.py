import numpy as np

from numpy.polynomial.chebyshev import chebval, chebfit


class calib_model(object):
    """ A class for modelling spectrophotometric calibration.

    Parameters
    ----------

    calib_dict : dictionary
        Contains the desired parameters for the calibration model.

    spectrum : array_like
        The spectral data to which the calibration model is applied.

    spectral_model : array_like
        The physical model which is being fitted to the data.
    """

    def __init__(self, calib_dict, spectrum, spectral_model):
        self.param = calib_dict
        self.y = spectrum[:, 1]
        self.y_err = spectrum[:, 2]
        self.y_model = spectral_model[:, 1]

        # Transform the spectral wavelengths to the interval (-1, 1).
        x = spectrum[:, 0]
        self.x = 2.*(x - (x[0] + (x[-1] - x[0])/2.))/(x[-1] - x[0])

        # Call the appropriate method to calculate the calibration.
        getattr(self, self.param["type"])()

    def polynomial_bayesian(self):
        """ Bayesian fitting of Chebyshev calibration polynomial. """

        coefs = []
        while str(len(coefs)) in list(self.param):
            coefs.append(self.param[str(len(coefs))])

        self.poly_coefs = np.array(coefs)
        self.model = chebval(self.x, coefs)

    def polynomial_max_like(self):
        order = int(self.param["order"])
        ratio = self.y_model/self.y
        errs = np.abs(self.y_err*self.y_model/self.y**2)
        coefs = chebfit(self.x, ratio, order, w=1./errs)

        self.poly_coefs = np.array(coefs)
        self.model = chebval(self.x, coefs)

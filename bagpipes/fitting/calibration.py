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
        self.wavs = spectrum[:, 0]

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

    def double_polynomial_bayesian(self):
        """ Bayesian fitting of Chebyshev calibration polynomial. """

        x_blue = self.wavs[self.wavs < self.param["wav_cut"]]
        x_red = self.wavs[self.wavs > self.param["wav_cut"]]

        self.x_blue = 2.*(x_blue - (x_blue[0] + (x_blue[-1] - x_blue[0])/2.))
        self.x_blue /= (x_blue[-1] - x_blue[0])

        self.x_red = 2.*(x_red - (x_red[0] + (x_red[-1] - x_red[0])/2.))
        self.x_red /= (x_red[-1] - x_red[0])

        blue_coefs = []
        red_coefs = []

        while "blue" + str(len(blue_coefs)) in list(self.param):
            blue_coefs.append(self.param["blue" + str(len(blue_coefs))])

        while "red" + str(len(red_coefs)) in list(self.param):
            red_coefs.append(self.param["red" + str(len(red_coefs))])

        self.blue_poly_coefs = np.array(blue_coefs)
        self.red_poly_coefs = np.array(red_coefs)

        model = np.zeros_like(self.x)
        model[self.wavs < self.param["wav_cut"]] = chebval(self.x_blue,
                                                           blue_coefs)

        model[self.wavs > self.param["wav_cut"]] = chebval(self.x_red,
                                                           red_coefs)

        self.model = model

    def polynomial_max_like(self):
        order = int(self.param["order"])

        mask = (self.y == 0.)

        ratio = self.y_model/self.y
        errs = np.abs(self.y_err*self.y_model/self.y**2)

        ratio[mask] = 0.
        errs[mask] = 9.9*10**99

        coefs = chebfit(self.x, ratio, order, w=1./errs)

        self.poly_coefs = np.array(coefs)
        self.model = chebval(self.x, coefs)

    def multi_polynomial_max_like(self):

        slice_order = int(self.param["slice_order"])
        n_slices = int(self.param["n_slices"])

        sect_length = (self.x[-1] - self.x[0])/n_slices

        poly = np.zeros_like(self.x)

        for i in range(n_slices):
            mask = (self.x >= self.x[0] + sect_length*i) & (self.x < self.x[0] + sect_length*(i+1))

            if i == n_slices - 1:
                mask = (self.x >= self.x[0] + sect_length*i) & (self.x <= self.x[0] + sect_length*(i+1))

            ratio = self.y_model[mask]/self.y[mask]
            errs = np.abs(self.y_err[mask]*self.y_model[mask]/self.y[mask]**2)

            coefs = chebfit(self.x[mask], ratio, slice_order, w=1./errs)
            model = chebval(self.x[mask], coefs)

            poly[mask] = model

            self.model = poly

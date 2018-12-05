import numpy as np
import george

from george import kernels


class noise_model(object):
    """ A class for modelling the noise properties of spectroscopic
    data, including correlated noise.

    Parameters
    ----------

    noise_dict : dictionary
        Contains the desired parameters for the noise model.

    inverse_variances : array_like
        The inverse of the variance for each data point.
    """

    def __init__(self, noise_dict, spectrum, spectral_model):
        self.param = noise_dict

        self.max_y = np.max(spectrum[:, 1])

        # Normalise the data in y by dividing througy by max value.
        self.y = spectrum[:, 1]/self.max_y
        self.y_err = spectrum[:, 2]/self.max_y
        self.y_model = spectral_model/self.max_y

        self.diff = self.y - self.y_model

        # Normalise the data in x.
        self.x = spectrum[:, 0] - spectrum[0, 0]
        self.x /= self.x[-1]

        if "type" in list(self.param):
            getattr(self, self.param["type"])()

        else:
            self.inv_var = 1./(self.max_y*self.y_err)**2
            self.corellated = False

    def white_scaled(self):
        """ A simple noise model with no covariances. Simply scales the
        input error spectrum by a constant factor. """

        self.inv_var = 1./(self.max_y*self.y_err*self.param["scaling"])**2
        self.corellated = False

    def gaussian_process(self):
        """ A GP noise model including various options for corellated
        noise and white noise (jitter term). """

        norm = self.param["norm"]
        length = self.param["length"]
        self.scaling = self.param["scaling"]

        kernel = norm**2*kernels.ExpSquaredKernel(length**2)
        self.gp = george.GP(kernel)
        self.gp.compute(self.x, self.y_err*self.scaling)

        self.corellated = True

    def mean(self):
        if self.corellated:
            return self.max_y*self.gp.predict(self.diff, self.x,
                                              return_cov=False)

        else:
            return np.zeros_like(self.x)

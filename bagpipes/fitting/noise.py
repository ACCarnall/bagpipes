import numpy as np

try:
    import george
    from george import kernels

except ImportError:
    pass


class noise_model(object):
    """ A class for modelling the noise properties of spectroscopic
    data, including correlated noise.

    Parameters
    ----------

    noise_dict : dictionary
        Contains the desired parameters for the noise model.

    galaxy : bagpipes.galaxy
        The galaxy object being fitted.

    spectral_model : array_like
        The physical model which is being fitted to the data.
    """

    def __init__(self, noise_dict, galaxy, spectral_model):
        self.param = noise_dict
        spectrum = np.copy(galaxy.spectrum)

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
        """ A simple variable noise model with no covariances. Scales
        the input error spectrum by a constant factor. """

        self.inv_var = 1./(self.max_y*self.y_err*self.param["scaling"])**2
        self.corellated = False

    def GP_exp_squared(self):
        """ A GP noise model including an exponenetial squared kernel
        for corellated noise and white noise (jitter term). """

        scaling = self.param["scaling"]

        norm = self.param["norm"]
        length = self.param["length"]

        kernel = norm**2*kernels.ExpSquaredKernel(length**2)
        self.gp = george.GP(kernel)
        self.gp.compute(self.x, self.y_err*scaling)

        self.corellated = True

    def GP_double_exp_squared(self):
        """ A GP noise model including a double exponenetial squared
        kernel for corellated noise and white noise (jitter term). """

        scaling = self.param["scaling"]

        norm1 = self.param["norm1"]
        length1 = self.param["length1"]

        norm2 = self.param["norm2"]
        length2 = self.param["length2"]

        kernel = (norm1**2*kernels.ExpSquaredKernel(length1**2)
                  + norm2**2*kernels.ExpSquaredKernel(length2**2))

        self.gp = george.GP(kernel)
        self.gp.compute(self.x, self.y_err*scaling)

        self.corellated = True

    def mean(self):
        if self.corellated:
            return self.max_y*self.gp.predict(self.diff, self.x,
                                              return_cov=False)

        else:
            return np.zeros_like(self.x)

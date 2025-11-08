import numpy as np

try:
    import george
    from george import kernels

except ImportError:
    pass

try:
    import celerite2
    from celerite2 import terms

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

    def GP_SHOTerm(self):
        """ 
        A GP noise model that uses celerite2's SHOTerm kernel 
        for corellated noise and white noise (jitter term). 
        This have been shown to be similar in behaviour to the
        squared exponential kernel but at least 100x faster.
        For more detail, see Leung et al. 2024
        (https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.4029L)
        The kernel is implemented through Dan Foreman-Mackey's celerite2
        For details, see https://celerite2.readthedocs.io/en/latest/ and
        http://adsabs.harvard.edu/abs/2018RNAAS...2a..31F
        """

        scaling = self.param["scaling"]

        # amplitude of the GP component
        norm = self.param["norm"]
        # the undamped period of the oscillator
        period = self.param["period"]
        # dampening quality factor Q
        Q = self.param["Q"]

        kernel = terms.SHOTerm(sigma=norm, rho=period, Q=Q)
        self.gp = celerite2.GaussianProcess(kernel)
        self.gp.compute(self.x, yerr=self.y_err*scaling)
        # renaming the method gp.log_likelihood to gp.lnlikelihood 
        # to be consistent with the old package george
        # so _lnlike_spec in fitted_model.py does not need changing
        self.gp.lnlikelihood = self.gp.log_likelihood

        self.corellated = True

    def mean(self):
        if self.corellated:
            return self.max_y*self.gp.predict(self.diff, self.x,
                                              return_cov=False)

        else:
            return np.zeros_like(self.x)

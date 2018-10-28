from __future__ import print_function, division, absolute_import

import numpy as np
import time

from copy import deepcopy
from numpy.polynomial.chebyshev import chebval, chebfit

from .prior import prior
from ..models.model_galaxy import model_galaxy


class fitted_model(object):
    """ Contains a model which is to be fitted to observational data.

    Parameters
    ----------

    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    fit_instructions : dict
        A dictionary containing instructions on the kind of model which
        should be fitted to the data.
    """

    def __init__(self, galaxy, fit_instructions):

        self.galaxy = galaxy
        self.fit_instructions = deepcopy(fit_instructions)
        self.model_components = deepcopy(fit_instructions)

        self._set_constants()
        self._process_fit_instructions()

        self.prior = prior(self.limits, self.pdfs, self.hyper_params)
        self.model_galaxy = None
        self.times = []

    def _process_fit_instructions(self):
        all_keys = []           # All keys in fit_instructions and subs
        all_vals = []           # All vals in fit_instructions and subs

        self.params = []        # Parameters to be fitted
        self.limits = []        # Limits for fitted parameter values
        self.pdfs = []          # Probability densities within lims
        self.hyper_params = []  # Hyperparameters of prior distributions
        self.mirror_pars = {}   # Params which mirror a fitted param

        # Flatten the input fit_instructions dictionary.
        for key in list(self.fit_instructions):
            if not isinstance(self.fit_instructions[key], dict):
                all_keys.append(key)
                all_vals.append(self.fit_instructions[key])

            else:
                for sub_key in list(self.fit_instructions[key]):
                    all_keys.append(key + ":" + sub_key)
                    all_vals.append(self.fit_instructions[key][sub_key])

        # Sort the resulting lists alphabetically by parameter name.
        indices = np.argsort(all_keys)
        all_vals = [all_vals[i] for i in indices]
        all_keys.sort()

        # Find parameters to be fitted and extract their priors.
        for i in range(len(all_vals)):
            if isinstance(all_vals[i], tuple):
                self.params.append(all_keys[i])
                self.limits.append(all_vals[i])  # Limits on prior.

                # Prior probability densities between these limits.
                prior_key = all_keys[i] + "_prior"
                if prior_key in list(all_keys):
                    self.pdfs.append(all_vals[all_keys.index(prior_key)])

                else:
                    self.pdfs.append("uniform")

                # Any hyper-parameters of these prior distributions.
                self.hyper_params.append({})
                for i in range(len(all_keys)):
                    if all_keys[i].startswith(prior_key + "_"):
                        hyp_key = all_keys[i][len(prior_key)+1:]
                        self.hyper_params[-1][hyp_key] = all_vals[i]

            # Find any parameters which mirror the value of a fit param.
            if all_vals[i] in all_keys:
                self.mirror_pars[all_keys[i]] = all_vals[i]

        # Find the dimensionality of the fit
        self.ndim = len(self.params)

    def _set_constants(self):
        """ Calculate constant factors used in the lnlike function. """

        if self.galaxy.spectrum_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.spectrum[:, 2]**2)
            self.K_spec = -0.5*np.sum(log_error_factors)
            self.N_spec = self.galaxy.spectrum.shape[0]
            self.inv_sigma_sq_spec = 1./self.galaxy.spectrum[:, 2]**2

        if self.galaxy.photometry_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.photometry[:, 2]**2)
            self.K_phot = -0.5*np.sum(log_error_factors)
            self.inv_sigma_sq_phot = 1./self.galaxy.photometry[:, 2]**2

    def lnlike(self, x, ndim=0, nparam=0):
        """ Returns the log-likelihood for a given parameter vector. """

        # Update the model_galaxy with the parameters from the sampler.
        self._update_model_components(x)

        if self.model_galaxy is None:
            self.model_galaxy = model_galaxy(self.model_components,
                                             filt_list=self.galaxy.filt_list,
                                             spec_wavs=self.galaxy.spec_wavs)

        self.model_galaxy.update(self.model_components)

        if "polynomial" in list(self.model_components):
            self._update_polynomial()

        # Return zero likelihood if SFH is older than the universe.
        if self.model_galaxy.sfh.unphysical:
            return -9.99*10**99

        lnlike = 0.

        if self.galaxy.spectrum_exists:
            lnlike += self._lnlike_spec()

        if self.galaxy.photometry_exists:
            lnlike += self._lnlike_phot()

        # Return zero likelihood if lnlike = nan (something went wrong).
        if np.isnan(lnlike):
            print("Bagpipes: lnlike was nan, replaced with zero probability.")
            return -9.99*10**99

        return lnlike

    def _lnlike_phot(self):
        """ Calculates the log-likelihood for photometric data. """

        diff = (self.galaxy.photometry[:, 1] - self.model_galaxy.photometry)**2
        self.chisq_phot = np.sum(diff*self.inv_sigma_sq_phot)

        return self.K_phot - 0.5*self.chisq_phot

    def _lnlike_spec(self):
        """ Calculates the log-likelihood for spectroscopic data.
        This includes options for fitting a spectral calibration
        polynomial and modelling problems with the error spectrum. """

        # Optionally divide the model by a polynomial for calibration.
        if "polynomial" in list(self.fit_instructions):
            diff = (self.galaxy.spectrum[:, 1]
                    - self.model_galaxy.spectrum[:, 1]/self.polynomial)**2

        else:
            diff = (self.galaxy.spectrum[:, 1]
                    - self.model_galaxy.spectrum[:, 1])**2

        # Optionally blow up the error spectrum by a factor sig_exp.
        sig_exp = 1.
        if "noise" in list(self.model_components):
            sig_exp = self.model_components["noise"]["sig_exp"]

        chisq_spec = np.sum(diff*self.inv_sigma_sq_spec)/sig_exp**2

        return self.K_spec - self.N_spec*np.log(sig_exp) - 0.5*chisq_spec

    def _update_model_components(self, param):
        """ Generates a model object with the current parameters. """

        # Substitute values of fit params from param into model_comp.
        for i in range(len(self.params)):
            split = self.params[i].split(":")
            if len(split) == 1:
                self.model_components[self.params[i]] = param[i]

            elif len(split) == 2:
                self.model_components[split[0]][split[1]] = param[i]

        # Set any mirror params to the value of the relevant fit param.
        for key in list(self.mirror_pars):
            split_par = key.split(":")
            split_val = self.mirror_pars[key].split(":")
            fit_val = self.model_components[split_val[0]][split_val[1]]
            self.model_components[split_par[0]][split_par[1]] = fit_val

    def _update_polynomial(self):
        """ Update the spectral calibration polynomial. """

        # Transform spec_wavs into interval (-1, 1).
        x = np.copy(self.galaxy.spec_wavs)
        x = 2.*(x - (x[0] + (x[-1] - x[0])/2.))/(x[-1] - x[0])

        # Get coefficients for the polynomial.
        if self.model_components["polynomial"]["type"] == "bayesian":
            coefs = []
            poly_dict = self.model_components["polynomial"]

            while str(len(coefs)) in list(poly_dict):
                coefs.append(poly_dict[str(len(coefs))])

        elif self.model_components["polynomial"]["type"] == "max_like":
            y = self.model_galaxy.spectrum[:, 1]/self.galaxy.spectrum[:, 1]
            n = int(self.fit_instructions["polynomial"]["order"])
            coefs = chebfit(x, y, n, w=self.inv_sigma_spec)

        self.polynomial = chebval(x, coefs)
        self.poly_coefs = np.array(coefs)

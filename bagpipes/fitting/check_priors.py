from __future__ import print_function, division, absolute_import

import numpy as np

from copy import deepcopy

from ..models.star_formation_history import star_formation_history
from ..models.model_galaxy import model_galaxy

from .prior import prior, dirichlet


class check_priors:

    def __init__(self, fit_instructions, filt_list=None, spec_wavs=None,
                 n_draws=10000, phot_units="ergscma"):

        self.fit_instructions = deepcopy(fit_instructions)
        self.model_components = deepcopy(fit_instructions)
        self.filt_list = filt_list
        self.spec_wavs = spec_wavs
        self.n_draws = n_draws

        self._process_fit_instructions()

        self.prior = prior(self.limits, self.pdfs, self.hyper_params)

        # Set up the model galaxy
        self._update_model_components(self.prior.sample())
        self.sfh = star_formation_history(self.model_components)
        self.model_galaxy = model_galaxy(self.model_components,
                                         filt_list=self.filt_list,
                                         spec_wavs=self.spec_wavs,
                                         phot_units=phot_units)

        self.samples = {}
        self.samples2d = np.zeros((self.n_draws, self.ndim))

        for i in range(self.n_draws):
            self.samples2d[i, :] = self.prior.sample()

        for i in range(self.ndim):
            self.samples[self.params[i]] = self.samples2d[:, i]

        self.get_basic_quantities()

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

            if all_vals[i] == "dirichlet":
                n = all_vals[all_keys.index(all_keys[i][:-6])]
                comp = all_keys[i].split(":")[0]
                for j in range(1, n):
                    self.params.append(comp + ":dirichletr" + str(j))
                    self.pdfs.append("uniform")
                    self.limits.append((0., 1.))
                    self.hyper_params.append({})

        # Find the dimensionality of the fit
        self.ndim = len(self.params)

    def _update_model_components(self, param):
        """ Generates a model object with the current parameters. """

        dirichlet_comps = []

        # Substitute values of fit params from param into model_comp.
        for i in range(len(self.params)):
            split = self.params[i].split(":")
            if len(split) == 1:
                self.model_components[self.params[i]] = param[i]

            elif len(split) == 2:
                if "dirichlet" in split[1]:
                    if split[0] not in dirichlet_comps:
                        dirichlet_comps.append(split[0])

                else:
                    self.model_components[split[0]][split[1]] = param[i]

        # Set any mirror params to the value of the relevant fit param.
        for key in list(self.mirror_pars):
            split_par = key.split(":")
            split_val = self.mirror_pars[key].split(":")
            fit_val = self.model_components[split_val[0]][split_val[1]]
            self.model_components[split_par[0]][split_par[1]] = fit_val

        # Deal with any Dirichlet distributed parameters.
        if len(dirichlet_comps) > 0:
            comp = dirichlet_comps[0]
            n_bins = 0
            for i in range(len(self.params)):
                split = self.params[i].split(":")
                if (split[0] == comp) and ("dirichlet" in split[1]):
                    n_bins += 1

            self.model_components[comp]["r"] = np.zeros(n_bins)

            j = 0
            for i in range(len(self.params)):
                split = self.params[i].split(":")
                if (split[0] == comp) and "dirichlet" in split[1]:
                    self.model_components[comp]["r"][j] = param[i]
                    j += 1

            tx = dirichlet(self.model_components[comp]["r"],
                           self.model_components[comp]["alpha"])

            self.model_components[comp]["tx"] = tx

    def get_basic_quantities(self):
        """Calculates basic prior quantities, these are fast as they
        are derived only from the SFH model, not the spectral model. """

        if "stellar_mass" in list(self.samples):
            return

        self._update_model_components(self.samples2d[0, :])
        self.sfh = star_formation_history(self.model_components)

        quantity_names = ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr",
                          "mass_weighted_age", "tform", "tquench"]

        for q in quantity_names:
            self.samples[q] = np.zeros(self.n_draws)

        self.samples["sfh"] = np.zeros((self.n_draws, self.sfh.ages.shape[0]))

        quantity_names += ["sfh"]

        for i in range(self.n_draws):
            param = self.samples2d[i, :]
            self._update_model_components(param)
            self.sfh.update(self.model_components)

            for q in quantity_names:
                self.samples[q][i] = getattr(self.sfh, q)

    def get_advanced_quantities(self):
        """Calculates advanced derived prior quantities, these are
        slower because they require the full model spectra. """

        if "spectrum_full" in list(self.samples):
            return

        all_names = ["photometry", "spectrum", "spectrum_full", "uvj",
                     "indices"]

        all_model_keys = dir(self.model_galaxy)
        quantity_names = [q for q in all_names if q in all_model_keys]

        for q in quantity_names:
            size = getattr(self.model_galaxy, q).shape[0]
            self.samples[q] = np.zeros((self.n_draws, size))

        for i in range(self.n_draws):
            param = self.samples2d[i, :]
            self._update_model_components(param)
            self.model_galaxy.update(self.model_components)

            for q in quantity_names:
                if q == "spectrum":
                    spectrum = getattr(self.model_galaxy, q)[:, 1]
                    self.samples[q][i] = spectrum
                    continue

                self.samples[q][i] = getattr(self.model_galaxy, q)

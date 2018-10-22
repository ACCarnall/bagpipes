from __future__ import print_function, division, absolute_import

import numpy as np
import os
import sys
import time
import warnings
import deepdish

from copy import deepcopy, copy
from scipy.special import erf, erfinv
from dynesty import NestedSampler
from dynesty.utils import resample_equal, simulate_run
from numpy.polynomial.chebyshev import chebval, chebfit

from . import utils
from . import config
from . import priors
from . import plotting

from .model_galaxy import model_galaxy


class fit_info_parser:
    """ A class which provides an interface between the fit_instructions
    and model_components dictionaries and samplers.

    Parameters
    ----------

    fit_instructions : dict
        A dictionary containing the details of the model to be fitted
        to the data.

    """

    def __init__(self, fit_instructions):

        # Model: contains a model galaxy
        self.model = None

        # fit_instructions: details of the model to be fitted.
        self.fit_info = deepcopy(fit_instructions)

        # fit_limits: list of tuples containing upper and lower bounds
        # for each parameter to be fit
        self.fit_limits = []

        # fit_params: list of the names of each parameter to be fit.
        self.fit_params = []

        # fixed_params: list of the names of each parameter to be fixed.
        self.fixed_params = []

        # fixed_values: list of the values of each fixed parameter.
        self.fixed_values = []

        # priors: A list of the prior on each fitted parameter.
        self.priors = []

        # Populate the previous four lists from fit_instructions
        self._process_fit_instructions()

    def _process_fit_instructions(self):
        """ Sets up the class by generating relevant variables from the
        input fit_instructions dictionary. """

        for key in list(self.fit_info):

            param = self.fit_info[key]

            # Checks for global parameters to be fitted.
            if isinstance(param, tuple):
                self.fit_limits.append(param)
                self.fit_params.append(key)

            # Checks for global parameters to be fixed.
            elif isinstance(param, (float, int)):

                self.fixed_values.append(param)
                self.fixed_params.append(key)

            # Checks for model components (dictionaries).
            elif isinstance(param, dict):
                for comp_key in list(param):

                    comp_param = param[comp_key]

                    # checks for component parameters to be fitted.
                    if isinstance(comp_param, tuple):
                        self.fit_limits.append(comp_param)
                        self.fit_params.append(key + ":" + comp_key)

                    # Checks for component parameters to be fixed.
                    elif isinstance(comp_param, (float, int, str)):
                        pr = [comp_key.endswith(p) for p in ["type", "_prior"]]
                        if not any(pr):
                            self.fixed_values.append(comp_param)
                            self.fixed_params.append(key + ":" + comp_key)

        self.ndim = len(self.fit_params)

        indices = np.argsort(self.fit_params)
        self.fit_limits = [self.fit_limits[i] for i in indices]
        self.fit_params.sort()

        indices = np.argsort(self.fixed_params)
        self.fixed_values = [self.fixed_values[i] for i in indices]
        self.fixed_params.sort()

        # Populate list of priors
        for fit_param in self.fit_params:

            sp = fit_param.split(":")
            level = len(sp)

            if level == 1 and sp[0] + "_prior" in list(self.fit_info):
                self.priors.append(self.fit_info[sp[0] + "_prior"])

            elif level == 2 and sp[1] + "_prior" in list(self.fit_info[sp[0]]):
                self.priors.append(self.fit_info[sp[0]][sp[1] + "_prior"])

            else:
                self.priors.append("uniform")

        """
        Sets the max_redshift parameter to just above the maximum fitted
        redshift in order to speed up model generation. This is a super
        awesome option, but the calculated model fluxes are slightly
        different with different values of max_redshift. This messes
        things up e.g. when fitting high S/N mocks. Needs more thought.
        """
        """
        if "redshift" in self.fit_params:
            index = self.fit_params.index("redshift")
            utils.max_redshift = self.fit_limits[index][1] + 0.05

        elif "redshift" in self.fixed_params:
            index = self.fixed_params.index("redshift")
            utils.max_redshift = self.fixed_values[index] + 0.05
        """

    def _prior_transform(self, cube, ndim=0, nparam=0):
        """ Transforms from the unit cube to prior volume using the
        functions in the priors file. """

        for i in range(self.ndim):

            hyper_params = {}

            for key in self.fixed_params:

                key_start = self.fit_params[i] + "_prior_"

                if key.startswith(key_start):
                    key_end = key[len(key_start):]
                    index = self.fixed_params.index(key)
                    hyper_params[key_end] = self.fixed_values[index]

            prior_func = getattr(priors, self.priors[i])
            cube[i] = prior_func(cube[i], self.fit_limits[i],
                                 hyper_params=hyper_params)

        return cube

    def _get_model_comp(self, param):
        """ Turns a parameter vector into a model_comp dict. """

        # Generate a model_comp dict and insert parameter values.
        model_comp = deepcopy(self.fit_info)

        for i in range(len(self.fit_params)):
            split = self.fit_params[i].split(":")
            if len(split) == 1:
                model_comp[self.fit_params[i]] = param[i]

            elif len(split) == 2:
                model_comp[split[0]][split[1]] = param[i]

        # finds any dependent parameters, which are set to the
        # values of the parameters they depend on.
        for i in range(len(self.fixed_values)):
            param = self.fixed_values[i]
            if isinstance(param, str) and param not in ["age_of_universe"]:
                split_par = self.fixed_params[i].split(":")
                split_val = self.fixed_values[i].split(":")
                fixed_val = model_comp[split_val[0]][split_val[1]]
                model_comp[split_par[0]][split_par[1]] = fixed_val

        # Find any parameters fixed to age of the Universe.
        for i in range(len(self.fixed_values)):
            if self.fixed_values[i] is "age_of_universe":
                age_at_z = np.interp(model_comp["redshift"],
                                     utils.z_array, utils.age_at_z)

                split_par = self.fixed_params[i].split(":")

                if len(split_par) == 0:
                    model_comp[split_par[0]] = age_at_z

                else:
                    model_comp[split_par[0]][split_par[1]] = age_at_z

        return model_comp

    def _get_model(self, param):
        """ Generates a model object with the current parameters. """

        self.model_comp = self._get_model_comp(param)

        if self.model is None:
            if self.galaxy.spectrum_exists:
                self.model = model_galaxy(self.model_comp,
                                          self.galaxy.filter_set.filt_list,
                                          spec_wavs=self.galaxy.spectrum[:, 0])

            else:
                self.model = model_galaxy(self.model_comp,
                                          self.galaxy.filter_set.filt_list)

        else:
            self.model.update(self.model_comp)

        if "polynomial" in list(self.model_comp):
            self._get_poly()

    def _get_poly(self):

        x = np.copy(self.model.spec_wavs)
        width = (x[-1] - x[0])
        x -= x[0] + width/2
        x /= width/2

        if self.model_comp["polynomial"]["type"] == "bayesian":
            coefs = []
            poly_dict = self.model_comp["polynomial"]

            while str(len(coefs)) in list(poly_dict):
                coefs.append(poly_dict[str(len(coefs))])

        elif self.model_comp["polynomial"]["type"] == "max_like":
            y = self.model.spectrum[:, 1]/self.galaxy.spectrum[:, 1]
            n = int(self.fit_info["polynomial"]["order"])
            coefs = chebfit(x, y, n, w=self.inv_sigma_spec)

        self.polynomial = chebval(x, coefs)
        self.poly_coefs = np.array(coefs)


class fit(fit_info_parser):
    """ Fit a model to the data contained in a galaxy object.

    Parameters
    ----------

    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    fit_instructions : dict
        A dictionary containing the details of the model to be fitted
        to the data.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    time_calls: bool - optional
        Defaults to False, if True prints the time taken for each like-
        lihood call.

    """

    def __init__(self, galaxy, fit_instructions, run=".", time_calls=False):

        utils.make_dirs()

        # run: the name of the set of fits this fit belongs to, used to
        # set the location of saved output within the pipes directory.
        self.run = run

        # galaxy: galaxy object, contains the data to be fitted.
        self.galaxy = galaxy

        # time_calls, whether to time the _get_lnlike function.
        self.time_calls = time_calls

        # post_path: where the posterior file should be saved.
        self.post_path = ("pipes/posterior/" + self.run + "/"
                          + self.galaxy.ID + ".h5")

        # Set up variables which will be used when calculating lnlike.
        self.K_phot, self.K_spec = 0., 0.
        self.N_spec, self.N_phot = 0., 0.
        self.chisq_spec, self.chisq_phot = 0., 0.

        # Calculate constant factors to be added to lnlike.
        if self.galaxy.spectrum_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.spectrum[:, 2]**2)
            self.K_spec = -0.5*np.sum(log_error_factors)
            self.N_spec = self.galaxy.spectrum.shape[0]
            self.inv_sigma_sq_spec = 1./self.galaxy.spectrum[:, 2]**2
            self.inv_sigma_spec = 1./self.galaxy.spectrum[:, 2]

        if self.galaxy.photometry_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.photometry[:, 2]**2)
            self.K_phot = -0.5*np.sum(log_error_factors)
            self.N_phot = self.galaxy.photometry.shape[0]
            self.inv_sigma_sq_phot = 1./self.galaxy.photometry[:, 2]**2

        # If there is already a saved posterior distribution load it.
        if os.path.exists(self.post_path):
            print("\nBagpipes: Existing posterior distribution and fit "
                  + "instructions loaded for object " + self.galaxy.ID + ".\n")

            self.posterior = deepdish.io.load(self.post_path)
            fit_info_parser.__init__(self, self.posterior["fit_instructions"])

            self._print_posterior()
            median = [self.posterior["median"][p] for p in self.fit_params]
            self._get_model(median)
            self._get_post_info()

        else:
            fit_info_parser.__init__(self, fit_instructions)

            self.posterior = {}
            self.posterior["fit_instructions"] = self.fit_info

        # Set up directories to contain the outputs.
        if self.run is not ".":
            if not os.path.exists("pipes/posterior/" + self.run):
                os.mkdir("pipes/posterior/" + self.run)

            if not os.path.exists("pipes/plots/" + self.run):
                os.mkdir("pipes/plots/" + self.run)

    def _fit_pmn(self, verbose=False, n_live=400):
        """ Run the pymultinest sampler. """

        import pymultinest as pmn

        name = "pipes/posterior/" + self.run + "/" + self.galaxy.ID + "-"

        pmn.run(self._get_lnlike, self._prior_transform, self.ndim,
                importance_nested_sampling=False, verbose=verbose,
                sampling_efficiency="model", n_live_points=n_live,
                outputfiles_basename=name)

        a = pmn.Analyzer(n_params=self.ndim, outputfiles_basename=name)

        s = a.get_stats()

        post_path = ("pipes/posterior/" + self.run + "/" + self.galaxy.ID
                     + "-post_equal_weights.dat")

        evidence_key = "nested sampling global log-evidence"

        # Add basic quantities to posterior.
        self.posterior["samples"] = np.loadtxt(post_path)[:, :-1]
        self.posterior["log_evidence"] = s[evidence_key]
        self.posterior["log_evidence_err"] = s[evidence_key + " error"]

        os.system("rm " + name + "*")

    def _fit_dynesty(self, verbose=False, n_live=400):
        """ Run the dynesty sampler. """
        if self.ndim <= 5:
            walk = 20

        elif self.ndim <= 10:
            walk = 25

        else:
            walk = 40

        self.sampler = NestedSampler(self._get_lnlike, self._prior_transform,
                                     self.ndim, nlive=n_live, bound="multi",
                                     sample="rwalk", walks=walk)

        self.sampler.run_nested(dlogz=0.01, print_progress=verbose)

        weights = np.exp(self.sampler.results.logwt
                         - self.sampler.results.logz[-1])

        weights /= weights.sum()

        post_eq = resample_equal(self.sampler.results.samples, weights)

        # Calculate error on the log_evidence.
        ev_arr = np.zeros(100)
        for i in range(100):
            ev_arr[i] = simulate_run(self.sampler.results).logz[-1]

        # Add basic quantities to posterior.
        self.posterior["samples"] = post_eq
        self.posterior["log_evidence"] = self.sampler.results.logz[-1]
        self.posterior["log_evidence_err"] = np.std(ev_arr)

    def fit(self, verbose=False, n_live=400, sampler="dynesty",
            calc_post=True, save_full_post=False):
        """ Fit the specified model to the input galaxy data using the
        dynesty or MultiNest algorithms.

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get progress updates from the sampler.

        n_live : int - optional
            Number of live points: reducing speeds up the code but may
            lead to unreliable results.

        sampler : string - optional
            Defaults to "dynesty" to use the Dynesty Python sampler.
            Can also be set to "pmn" to use PyMultiNest, however this
            requires the MultiNest Fortran libraries to be installed.

        calc_post : bool - optional
            If True (default), calculates a bunch of useful posterior
            quantities. These are needed for making plots, but things
            can be speeded up by setting this to False.

        save_full_post : bool - optional
            If True, saves all posterior information to disk instead of
            just the bare minimum needed to reconstruct the fit. This is
            set to False by default to save disk space.
        """

        if "samples" in list(self.posterior):
            print("\nBagpipes: Posterior already loaded from "
                  + self.post_path + ". To start from scratch, delete this "
                  + "file or change run.\n")

            if save_full_post:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    deepdish.io.save(self.post_path, self.posterior)

            return

        sampler_names = {"pmn": "PyMultiNest", "dynesty": "Dynesty"}
        print("\nBagpipes: fitting object " + self.galaxy.ID
              + " with " + sampler_names[sampler] + "\n")

        start_time = time.time()

        if sampler == "pmn":
            self._fit_pmn(verbose=verbose, n_live=n_live)

        elif sampler == "dynesty":
            self._fit_dynesty(verbose=verbose, n_live=n_live)

        # Calculate further basic posterior quantities.
        self.posterior["median"] = {}
        self.posterior["confidence_interval"] = {}

        for j in range(len(self.fit_params)):
            param = self.fit_params[j]

            post_median = np.median(self.posterior["samples"][:, j])
            conf_int = [np.percentile(self.posterior["samples"][:, j], 16),
                        np.percentile(self.posterior["samples"][:, j], 84)]

            self.posterior["median"][param] = post_median
            self.posterior["confidence_interval"][param] = conf_int

        # Print the results
        runtime = time.time() - start_time
        print("\nBagpipes: fitting complete in " + str("%.1f" % runtime)
              + " seconds.\n")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deepdish.io.save(self.post_path, self.posterior)

        if calc_post:
            self._get_post_info()

        if save_full_post:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                deepdish.io.save(self.post_path, self.posterior)

        self._print_posterior()

    def _print_posterior(self):
        """ Print the 16th, 50th, 84th percentiles of the posterior. """
        print("{:<25}".format("Parameter")
              + "{:>31}".format("Posterior percentiles"))

        print("{:<25}".format(""),
              "{:>10}".format("16th"),
              "{:>10}".format("50th"),
              "{:>10}".format("84th")
              )

        print("-"*58)

        for par in self.fit_params:
            conf_int = self.posterior["confidence_interval"][par]

            print("{:<25}".format(par),
                  "{:>10.3f}".format(conf_int[0]),
                  "{:>10.3f}".format(self.posterior["median"][par]),
                  "{:>10.3f}".format(conf_int[1])
                  )

        print("\n")

    def _get_lnlike_spec(self):

        if not self.galaxy.spectrum_exists:
            return 0.

        if "polynomial" in self.fit_info:
            diff = (self.galaxy.spectrum[:, 1]
                    - self.model.spectrum[:, 1]/self.polynomial)

        else:
            diff = (self.galaxy.spectrum[:, 1] - self.model.spectrum[:, 1])

        if "noise" in list(self.fit_info):
            if "mu" in list(self.model_comp["noise"]):
                median_y_val = np.median(self.galaxy.spectrum[:, 1])
                noise_mu = self.model_comp["noise"]["mu"]*median_y_val
                noise_sig = self.model_comp["noise"]["sigma"]*median_y_val
                noise_prob = self.model_comp["noise"]["prob"]

            if "sig_exp" in list(self.model_comp["noise"]):
                sig_exp = self.model_comp["noise"]["sig_exp"]

            else:
                sig_exp = 1.

            model_lnlike = (- 0.5*np.log(sig_exp**2)
                            - 0.5*np.log(2*np.pi*self.galaxy.spectrum[:, 2]**2)
                            - 0.5*(diff/self.galaxy.spectrum[:, 2]/sig_exp)**2)

            if "mu" in list(self.model_comp["noise"]):
                combined_var = (noise_sig**2
                                + (self.galaxy.spectrum[:, 2]*sig_exp)**2)

                noise_lnlike = (- 0.5*np.log(2*np.pi*combined_var)
                                - 0.5*(self.galaxy.spectrum[:, 1]
                                       - noise_mu)**2/combined_var
                                + np.log(noise_prob))

                model_lnlike += np.log(1. - noise_prob)

                combined_lnlike = np.logaddexp(model_lnlike, noise_lnlike)

            else:
                combined_lnlike = model_lnlike

            lnlike_spec = np.sum(combined_lnlike)

            if "mu" in list(self.model_comp["noise"]):
                self.outlier_probs = np.exp(noise_lnlike - combined_lnlike)

            self.chisq_spec = np.sum(diff**2*self.inv_sigma_sq_spec)

        else:
            self.chisq_spec = np.sum(diff**2*self.inv_sigma_sq_spec)

            lnlike_spec = -0.5*self.chisq_spec

        return lnlike_spec

    def _get_polynomial_prior(self):
        if "prior_width" not in list(self.fit_info["polynomial"]):
            return 0.

        width = self.model_comp["polynomial"]["prior_width"]
        diff = self.polynomial - 1.

        return -0.5*np.sum((diff/width)**2)

    def _get_lnlike_phot(self):

        if not self.galaxy.photometry_exists:
            return 0.

        diff = (self.galaxy.photometry[:, 1] - self.model.photometry)
        self.chisq_phot = np.sum(diff**2*self.inv_sigma_sq_phot)

        return self.K_phot - 0.5*self.chisq_phot

    def _get_lnlike(self, x, ndim=0, nparam=0):
        """ Returns the log-probability for a given model sfh and
        parameter vector x. """

        if self.time_calls:
            time0 = time.time()

        self._get_model(x)

        # Return zero likelihood if the model is older than the universe
        if self.model.sfh.unphysical:
            return -9.99*10**99

        lnlike_spec = self._get_lnlike_spec()
        lnlike_phot = self._get_lnlike_phot()

        lnlike = lnlike_phot + lnlike_spec

        # Apply a separate prior for maximum likelihood prior fitting
        if "polynomial" in self.fit_info:
            lnlike += self._get_polynomial_prior()

        # Return zero likelihood if lnlike is nan (shouldn't happen)
        if np.isnan(lnlike):
            print("Bagpipes: lnlike was nan, replaced with zero probability.")
            return -9.99*10**99

        if self.time_calls:
            print(time.time() - time0)

        return lnlike

    def _get_post_info(self, max_size=100):
        """ Calculates posterior quantities which require models to be
        re-generated. For all posterior quantities the first index runs
        over the posterior samples. """

        for j in range(len(self.fit_params)):
            param_samples = self.posterior["samples"][:, j]
            self.posterior[self.fit_params[j]] = param_samples

        # The posterior can be huge, so select a subsample of models to
        # calculate, the number of samples is determined by max_size.
        n_post_full = self.posterior["samples"].shape[0]

        if n_post_full > max_size:
            n_post = max_size
            chosen = np.random.choice(np.arange(n_post_full), size=n_post,
                                      replace=False)
        else:
            n_post = n_post_full
            chosen = np.arange(n_post)

        self.posterior["chosen_samples"] = chosen

        # Set up the structure of the posterior dictionary.
        self.posterior["lnlike"] = np.zeros(n_post)

        self.posterior["mass"] = {}
        self.posterior["mass"]["total"] = {}
        self.posterior["mass"]["total"]["living"] = np.zeros(n_post)
        self.posterior["mass"]["total"]["formed"] = np.zeros(n_post)

        for comp in self.model.sfh.sfh_components:
            self.posterior["mass"][comp] = {}
            self.posterior["mass"][comp]["living"] = np.zeros(n_post)
            self.posterior["mass"][comp]["formed"] = np.zeros(n_post)

        self.posterior["sfh"] = np.zeros((n_post,
                                          self.model.sfh.ages.shape[0]))

        self.posterior["sfr"] = np.zeros(n_post)
        self.posterior["ssfr"] = np.zeros(n_post)
        self.posterior["mwa"] = np.zeros(n_post)
        self.posterior["tmw"] = np.zeros(n_post)
        self.posterior["tquench"] = np.zeros(n_post)
        self.posterior["tau_q"] = np.zeros(n_post)
        self.posterior["fwhm_sf"] = np.zeros(n_post)
        self.posterior["nsfr"] = np.zeros(n_post)

        self.posterior["age_of_universe"] = np.zeros(n_post)

        if self.galaxy.photometry_exists:
            self.posterior["UVJ"] = np.zeros((n_post, 3))

        len_spec_full = self.model.spectrum_full.shape[0]
        self.posterior["spectrum_full"] = np.zeros((n_post, len_spec_full))

        if self.galaxy.photometry_exists:
            len_phot = self.model.photometry.shape[0]
            self.posterior["photometry"] = np.zeros((n_post, len_phot))

        if self.galaxy.spectrum_exists:
            len_spec = self.model.spectrum.shape[0]
            self.posterior["spectrum"] = np.zeros((n_post, len_spec))

        if "polynomial" in list(self.fit_info):
            self.posterior["polynomial"] = np.zeros((n_post, len_spec))
            self.posterior["poly_coefs"] = np.zeros((n_post,
                                                     self.poly_coefs.shape[0]))

        if "noise" in list(self.fit_info):
            if "mu" in list(self.fit_info["noise"]):
                self.posterior["outlier_probs"] = np.zeros((n_post, len_spec))

        if self.model.nebular:
            self.posterior["line_fluxes"] = {}
            for name in config.line_names:
                self.posterior["line_fluxes"][name] = np.zeros(n_post)

        # For each point in the posterior, generate a model and extract
        # relevant quantities.
        for i in range(n_post):

            lnlike = self._get_lnlike(self.posterior["samples"][chosen[i], :])

            self.posterior["lnlike"][i] = lnlike

            sfh = self.model.sfh
            model_mass = sfh.mass
            post_mass = self.posterior["mass"]

            post_mass["total"]["living"][i] = model_mass["total"]["living"]
            post_mass["total"]["formed"][i] = model_mass["total"]["formed"]
            for comp in self.model.sfh.sfh_components:
                post_mass[comp]["living"][i] = model_mass[comp]["living"]
                post_mass[comp]["formed"][i] = model_mass[comp]["formed"]

            self.posterior["sfh"][i, :] = sfh.sfr["total"]
            self.posterior["sfr"][i] = sfh.sfr_100myr
            self.posterior["mwa"][i] = 10**-9*sfh.mass_weighted_age
            self.posterior["age_of_universe"][i] = sfh.age_of_universe

            # Calculate time of quenching as defined in Carnall (2017)
            mass_contrib = sfh.sfr["total"]*sfh.age_widths
            prog_masses = np.cumsum(mass_contrib[::-1])[::-1]
            tunivs = sfh.age_of_universe - sfh.ages

            prog_masses = prog_masses[:np.argmax(tunivs < 0)]
            sfrs = sfh.sfr["total"][:np.argmax(tunivs < 0)]
            tunivs = tunivs[:np.argmax(tunivs < 0)]
            mean_sfrs = prog_masses/tunivs
            self.posterior["nsfr"][i] = sfrs[0]/mean_sfrs[0]

            if sfrs[0] > 0.1*mean_sfrs[0]:
                self.posterior["tquench"][i] = 999.

            else:
                quench_ind = np.argmax(sfrs > 0.1*mean_sfrs)
                self.posterior["tquench"][i] = 10**-9*(sfh.age_of_universe
                                                       - sfh.ages[quench_ind])

            end = np.argmax(sfrs > 0.5*sfrs.max())
            start = sfrs.shape[0] - np.argmax(sfrs[::-1] > 0.5*sfrs.max()) - 1
            self.posterior["fwhm_sf"][i] = 10**-9*(tunivs[end] - tunivs[start])

            if self.galaxy.photometry_exists:
                self.posterior["UVJ"][i, :] = self.model.uvj

            self.posterior["spectrum_full"][i, :] = self.model.spectrum_full

            if self.galaxy.photometry_exists:
                self.posterior["photometry"][i, :] = self.model.photometry

            if self.model.nebular:
                for name in config.line_names:
                    line_flux = self.model.line_fluxes[name]
                    self.posterior["line_fluxes"][name][i] = line_flux

            if self.galaxy.spectrum_exists:
                self.posterior["spectrum"][i, :] = self.model.spectrum[:, 1]

            if "polynomial" in list(self.fit_info):
                self.posterior["polynomial"][i, :] = self.polynomial
                self.posterior["poly_coefs"][i, :] = self.poly_coefs

        if "noise" in list(self.fit_info):
            if "mu" in list(self.fit_info["noise"]):
                self.posterior["outlier_probs"][i, :] = self.outlier_probs

        self.posterior["ssfr"] = np.log10(self.posterior["sfr"]
                                          / post_mass["total"]["living"])

        self.posterior["tmw"] = (self.posterior["age_of_universe"]*10**-9
                                 - self.posterior["mwa"])

        self.posterior["tau_q"] = ((self.posterior["tquench"]
                                    - self.posterior["tmw"])
                                   / self.posterior["tquench"])

        # Extract parameters associated with the maximum likelihood solution.
        self.posterior["maximum_likelihood"] = {}
        best_params = []

        for i in range(self.ndim):
            samples = self.posterior["samples"][:, i]
            best_param = samples[np.argmax(self.posterior["lnlike"])]
            best_params.append(best_param)

        self._get_lnlike(best_params, self.ndim, self.ndim)

        min_chisq = 0.
        min_chisq_red = 0.
        self.ndof = -self.ndim

        if self.galaxy.spectrum_exists:
            min_chisq += self.chisq_spec
            self.ndof += self.galaxy.spectrum.shape[0]

        if self.galaxy.photometry_exists:
            min_chisq += self.chisq_phot
            self.ndof += self.galaxy.photometry.shape[0]

        min_chisq_red = min_chisq/float(self.ndof)

        self.posterior["min_chisq"] = min_chisq
        self.posterior["min_chisq_reduced"] = min_chisq_red

        self.posterior["spectrum_full_wavs_rest"] = self.model.wavelengths
        self.posterior["sfh_ages"] = self.model.sfh.ages

        if self.galaxy.photometry_exists:
            self.posterior["eff_wavs"] = self.model.filter_set.eff_wavs

        if self.galaxy.spectrum_exists:
            self.posterior["spectrum_wavs_obs"] = self.model.spec_wavs

    def plot_fit(self, show=False, save=True):
        """ Make a plot of the input data and fitted posteriors. """
        return plotting.plot_fit(self, show=show, save=save)

    def plot_corner(self, show=False, save=True):
        """ Make a corner plot showing the posterior distributions of
        the fitted parameters. """
        return plotting.plot_corner(self, show=show, save=save)

    def plot_sfh(self, show=False, save=True):
        """ Make a plot of the star-formation history posterior. """
        return plotting.plot_sfh_post(self, show=show, save=save)

    def plot_poly(self, show=False, save=True):
        """ Make a plot showing the posterior distribution for the
        fitted polynomial. """

        if "polynomial" not in list(self.posterior):
            print("Bagpipes: No polynomial to plot.")
            return

        return plotting.plot_poly(self, show=show, save=save)

    def plot_1d_posterior(self, show=False, save=True):
        """ Make a plot of the 1d posterior distributions. """
        plotting.plot_1d_distributions(self, show=show, save=save)

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

from . import utils
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
            elif isinstance(param, float):
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
                    elif isinstance(comp_param, (float, str)):
                        if (comp_key is not "type"
                                and comp_key[-6:] != "_prior"):

                            self.fixed_values.append(comp_param)
                            self.fixed_params.append(key + ":" + comp_key)

        self.ndim = len(self.fit_params)

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
            if isinstance(param, str) and param is not "age_of_universe":
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
                                          self.galaxy.filt_list,
                                          spec_wavs=self.galaxy.spectrum[:, 0])

            else:
                self.model = model_galaxy(self.model_comp,
                                          self.galaxy.filt_list)

        else:
            self.model.update(self.model_comp)


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

    """

    def __init__(self, galaxy, fit_instructions, run="."):

        fit_info_parser.__init__(self, fit_instructions)

        utils.make_dirs()

        # run: the name of the set of fits this fit belongs to, used to
        # set the location of saved output within the pipes directory.
        self.run = run

        # galaxy: galaxy object, contains the data to be fitted.
        self.galaxy = galaxy

        # post_path: where the posterior file should be saved.
        self.post_path = ("pipes/posterior/" + self.run + "/"
                          + self.galaxy.ID + ".h5")

        # posterior: Will be used to store posterior samples.
        self.posterior = {}

        # If there is already a saved posterior distribution load it.
        if os.path.exists(self.post_path):
            print("\nBagpipes: Existing posterior distribution loaded for"
                  + " object " + self.galaxy.ID + ".\n")

            self.posterior = deepdish.io.load(self.post_path)
            self._print_posterior()
            post_med = [self.posterior["median"][p] for p in self.fit_params]
            self._get_model(post_med)

        # Set up directories to contain the outputs.
        if self.run is not ".":
            if not os.path.exists("pipes/posterior/" + self.run):
                os.mkdir("pipes/posterior/" + self.run)

            if not os.path.exists("pipes/plots/" + self.run):
                os.mkdir("pipes/plots/" + self.run)

        # Set up variables which will be used when calculating lnprob.
        self.K_phot, self.K_spec = 0., 0.
        self.N_spec, self.N_phot = 0., 0.
        self.hyp_spec, self.hyp_phot = 1., 1.
        self.chisq_spec, self.chisq_phot = 0., 0.

        # Calculate constant factors to be added to lnprob.
        if self.galaxy.spectrum_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.spectrum[:, 2]**2)
            self.K_spec = -0.5*np.sum(log_error_factors)
            self.N_spec = self.galaxy.spectrum.shape[0]
            self.inv_sigma_sq_spec = 1./self.galaxy.spectrum[:, 2]**2

        if self.galaxy.photometry_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.photometry[:, 2]**2)
            self.K_phot = -0.5*np.sum(log_error_factors)
            self.N_phot = self.galaxy.photometry.shape[0]
            self.inv_sigma_sq_phot = 1./self.galaxy.photometry[:, 2]**2

    def _fit_pmn(self, verbose=False, n_live=400):
        """ Run the pymultinest sampler. """

        import pymultinest as pmn

        name = "pipes/posterior/" + self.run + "/" + self.galaxy.ID + "-"

        pmn.run(self._get_lnprob, self._prior_transform, self.ndim,
                importance_nested_sampling=False, verbose=verbose,
                sampling_efficiency="model", n_live_points=n_live,
                outputfiles_basename=name)

        a = pmn.Analyzer(n_params=self.ndim, outputfiles_basename=name,
                         verbose=False)

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

        self.sampler = NestedSampler(self._get_lnprob, self._prior_transform,
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

    def fit(self, verbose=False, n_live=400,
            sampler="dynesty", time_calls=False):
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

        time_calls : bool - optional
            If True, prints the time taken for each likelihood call.
        """
        if time_calls:
            self.time_calls = True
        else:
            self.time_calls = False


        if "samples" in list(self.posterior):
            print("\nBagpipes: Posterior already loaded from " + self.post_path
                  + "\nBagpipes: To start from scratch, delete this "
                  + "file or change run.\n")

            return

        samp_names = {"pmn": "PyMultiNest", "dynesty": "Dynesty"}
        print("\nBagpipes: fitting object " + self.galaxy.ID
              + " with " + samp_names[sampler] + "\n")

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
            param_samples = self.posterior["samples"][:, j]

            self.posterior[param] = param_samples

            post_median = np.median(self.posterior["samples"][:, j])
            conf_int = [np.percentile(self.posterior["samples"][:, j], 16),
                        np.percentile(self.posterior["samples"][:, j], 84)]

            self.posterior["median"][param] = post_median
            self.posterior["confidence_interval"][param] = conf_int

        # Print the results
        runtime = time.time() - start_time
        print("\nBagpipes: fitting complete in " + str("%.1f" % runtime)
              + " seconds.\n")

        self._get_post_info()

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

    def _get_lnprob(self, x, ndim=0, nparam=0):
        """ Returns the log-probability for a given model sfh and
        parameter vector x. """

        if self.time_calls:
            time0 = time.time()

        self._get_model(x)

        # If the age of any model component is greater than the age of
        # the Universe, return zero probability.
        if self.model.sfh.unphysical:
            return -9.99*10**99

        if "hypspec" in list(self.model_comp):
            self.hyp_spec = self.model_comp["hypspec"]

        if "hypphot" in list(self.model_comp):
            self.hyp_phot = self.model_comp["hypphot"]

        if self.galaxy.spectrum_exists:
            diff = (self.galaxy.spectrum[:, 1] - self.model.spectrum[:, 1])
            self.chisq_spec = np.sum(diff**2*self.inv_sigma_sq_spec)

        if self.galaxy.photometry_exists:
            diff = (self.galaxy.photometry[:, 1] - self.model.photometry)
            self.chisq_phot = np.sum(diff**2*self.inv_sigma_sq_phot)

        lnprob = (self.K_phot + self.K_spec
                  + 0.5*self.N_spec*np.log(self.hyp_spec)
                  + 0.5*self.N_phot*np.log(self.hyp_phot)
                  - 0.5*self.hyp_phot*self.chisq_phot
                  - 0.5*self.hyp_spec*self.chisq_spec)

        # Catch any failures to generate a model and return zero prob.
        if np.isnan(lnprob):
            return -9.99*10**99

        if self.time_calls:
            print(time.time() - time0)

        return lnprob

    def _get_post_info(self, max_size=500):
        """ Calculates posterior quantities which require models to be
        re-generated. For all posterior quantities the first index runs
        over the posterior samples. """

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
        self.posterior["lnprob"] = np.zeros(n_post)

        self.posterior["mass"] = {}
        self.posterior["mass"]["total"] = {}
        self.posterior["mass"]["total"]["living"] = np.zeros(n_post)
        self.posterior["mass"]["total"]["formed"] = np.zeros(n_post)
        for comp in self.model.sfh_components:
            self.posterior["mass"][comp] = {}
            self.posterior["mass"][comp]["living"] = np.zeros(n_post)
            self.posterior["mass"][comp]["formed"] = np.zeros(n_post)

        self.posterior["sfh"] = np.zeros((n_post,
                                          self.model.sfh.ages.shape[0]))
        self.posterior["sfr"] = np.zeros(n_post)
        self.posterior["ssfr"] = np.zeros(n_post)
        self.posterior["mwa"] = np.zeros(n_post)
        self.posterior["tmw"] = np.zeros(n_post)
        self.posterior["UVJ"] = np.zeros((n_post, 3))

        len_spec_full = self.model.spectrum_full.shape[0]
        self.posterior["spectrum_full"] = np.zeros((n_post, len_spec_full))

        if self.galaxy.photometry_exists:
            len_phot = self.model.photometry.shape[0]
            self.posterior["photometry"] = np.zeros((n_post, len_phot))

        if self.galaxy.spectrum_exists:
            len_spec = self.model.spectrum.shape[0]
            self.posterior["spectrum"] = np.zeros((n_post, len_spec))

        if self.model.polynomial is not None:
            self.posterior["polynomial"] = np.zeros((n_post, len_spec))

        if self.model.nebular_on:
            self.posterior["line_fluxes"] = {}
            for name in utils.line_names:
                self.posterior["line_fluxes"][name] = np.zeros(n_post)

        # For each point in the posterior, generate a model and extract
        # relevant quantities.
        for i in range(n_post):

            lnprob = self._get_lnprob(self.posterior["samples"][chosen[i], :])

            self.posterior["lnprob"][i] = lnprob

            model_mass = self.model.sfh.mass
            post_mass = self.posterior["mass"]

            post_mass["total"]["living"][i] = model_mass["total"]["living"]
            post_mass["total"]["formed"][i] = model_mass["total"]["living"]
            for comp in self.model.sfh_components:
                post_mass[comp]["living"][i] = model_mass[comp]["living"]
                post_mass[comp]["formed"][i] = model_mass[comp]["formed"]

            self.posterior["sfh"][i, :] = self.model.sfh.sfr["total"]
            self.posterior["sfr"][i] = self.model.sfh.sfr_100myr
            self.posterior["mwa"][i] = 10**-9*self.model.sfh.mass_weighted_age
            self.posterior["tmw"][i] = (self.model.sfh.age_of_universe*10**-9
                                        - self.posterior["mwa"][i])

            self.posterior["UVJ"][i, :] = self.model.get_restframe_UVJ()

            self.posterior["spectrum_full"][i, :] = self.model.spectrum_full

            if self.galaxy.photometry_exists:
                self.posterior["photometry"][i, :] = self.model.photometry

            if self.model.nebular_on:
                for name in utils.line_names:
                    line_flux = self.model.line_fluxes[name]
                    self.posterior["line_fluxes"][name][i] = line_flux

            if self.galaxy.spectrum_exists:
                self.posterior["spectrum"][i, :] = self.model.spectrum[:, 1]

            if self.model.polynomial is not None:
                self.posterior["polynomial"][i, :] = self.model.polynomial

        self.posterior["ssfr"] = np.log10(self.posterior["sfr"]
                                          / post_mass["total"]["living"])

        # Extract parameters associated with the maximum likelihood solution.
        self.posterior["maximum_likelihood"] = {}
        best_params = []

        for i in range(self.ndim):
            samples = self.posterior[self.fit_params[i]]
            best_param = samples[np.argmax(self.posterior["lnprob"])]
            best_params.append(best_param)

        self._get_lnprob(best_params, self.ndim, self.ndim)

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deepdish.io.save(self.post_path, self.posterior)

    def plot_fit(self, show=False, save=True):
        """ Make a plot of the input data and fitted posteriors. """
        return plotting.plot_fit(self, show=show)

    def plot_corner(self, show=False):
        """ Make a corner plot showing the posterior distributions of
        the fitted parameters. """
        return plotting.plot_corner(self, show=show)

    def plot_sfh(self, show=True):
        """ Make a plot of the star-formation history posterior. """
        return plotting.plot_sfh_post(self, show=show)

    def plot_poly(self, style="percentiles", show=True):
        """ Make a plot showing the posterior distribution for the
        fitted polynomial. """

        if "polynomial" not in list(self.posterior):
            print("Bagpipes: No polynomial to plot.")
            return

        return plotting.plot_poly(self, style=style, show=show)

    def plot_1d_posterior(self, show=False, save=True):
        plotting.plot_1d_distributions(self, show=show, save=save)


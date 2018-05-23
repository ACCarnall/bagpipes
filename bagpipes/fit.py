from __future__ import print_function, division, absolute_import

import numpy as np
import os
import sys
import corner
import matplotlib.pyplot as plt
import time
import dynesty

from matplotlib import gridspec
from copy import deepcopy, copy
from scipy.special import erf, erfinv

try:
    import pymultinest as pmn

except:
    print("Bagpipes: MultiNest/PyMultiNest not installed, if you want fitting you'll need to set the sampler keyword argument to dynesty.")

from .utils import *
from .model_galaxy import Model_Galaxy

from matplotlib import rc
rc('text', usetex=True)

class Fit:

    """ Fit a model to the data contained in a Galaxy object.

    Parameters
    ----------

    Galaxy : bagpipes.Galaxy
        A Galaxy object containing the photomeric and/or spectroscopic data you wish to fit.

    fit_instructions : dict
        A dictionary containing the details of the model to be fitted to the data.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for fitting more than one model configuration to the same data.

    """

    def __init__(self, Galaxy, fit_instructions, run="."):

        make_dirs()

        # Model: contains a model galaxy 
        self.Model = None

        # run: the name of the set of fits this fit belongs to, used for plot/pmn output saving location
        self.run = run

        # Galaxy: The Galaxy object passed to Fit, contains the data to be fit.
        self.Galaxy = Galaxy

        # fit_instructions: Contains the instructions to build the model used to fit the data
        self.fit_instructions = deepcopy(fit_instructions)

        # fit_limits: A list of tuples containing the upper and lower bounds on each fit parameter
        self.fit_limits = []

        # fit_params: A list of the names of each parameter to be fit
        self.fit_params = []

        # fixed_params: A list of the names of each parameter which remains fixed
        self.fixed_params = []

        # fixed_values: A list of the values at which each fixed parameter is fixed
        self.fixed_values = []

        # priors: A list of the functional forms of the priors on each fit_parameter
        self.priors = []

        # Populate the previous four lists using the fit_instructions dictionary
        self.process_fit_instructions()

        # posterior: Will be used to store posterior samples.
        self.posterior = {}
        
        # extra_models: a list into which extra model items will be placed if the galaxy object has more thna one spectrum
        #if self.Galaxy.no_of_spectra > 1:
        #    self.extra_models = []
        #    for i in range(self.Galaxy.no_of_spectra-1):
        #        self.extra_append(None)
        
        #Set up directories to contain the outputs, if they don't already exist
        if self.run is not ".":

            if not os.path.exists(working_dir + "/pipes/pmn_chains/" + self.run):
                os.mkdir(working_dir + "/pipes/pmn_chains/" + self.run)

            if not os.path.exists(working_dir + "/pipes/plots/" + self.run):
                os.mkdir(working_dir + "/pipes/plots/" + self.run)

        # Set up a bunch of variables which will be used when calculating the lnprob values.
        self.K_phot, self.K_spec = 0., 0.
        self.N_spec, self.N_phot =  0., 0.
        self.hyp_spec, self.hyp_phot = 1., 1.
        self.chisq_spec, self.chisq_phot = 0., 0.

        if self.Galaxy.spectrum_exists == True:
            self.K_spec = -0.5*np.sum(np.log(2*np.pi*self.Galaxy.spectrum[:,2]**2))
            self.N_spec = self.Galaxy.spectrum.shape[0]

        if self.Galaxy.photometry_exists == True:
            self.K_phot = -0.5*np.sum(np.log(2*np.pi*self.Galaxy.photometry[:,2]**2))
            self.N_phot = self.Galaxy.photometry.shape[0]
        """
        # Set up corresponding variables for any extra spectra 
        if self.Galaxy.no_of_spectra > 1:
            self.extra_K_spec = np.zeros(self.Galaxy.no_of_spectra-1)
            self.extra_N_spec = np.zeros(self.Galaxy.no_of_spectra-1)
            self.hyp_extra_spec = np.zeros(self.Galaxy.no_of_spectra-1) + 1.
            self.chisq_extra_spec = np.zeros(self.Galaxy.no_of_spectra-1)

            for i in range(self.Galaxy.no_of_spectra-1):
                self.extra_K_spec[i] = -0.5*np.sum(np.log(2*np.pi*self.Galaxy.extra_spectra[i][:,2]**2))
                self.extra_N_spec[i] = self.Galaxy.extra_spectra[i].shape[0]
        """



    def process_fit_instructions(self):
        # Sets up the class by generating relevant variables from the input fit_instructions dictionary.

        for key in list(self.fit_instructions):
            if isinstance(self.fit_instructions[key], tuple):
                self.fit_limits.append(self.fit_instructions[key])
                self.fit_params.append(key)
                
            elif isinstance(self.fit_instructions[key], float):
                self.fixed_values.append(self.fit_instructions[key])
                self.fixed_params.append(key)
                
            elif isinstance(self.fit_instructions[key], dict):
                for sfh_comp_key in list(self.fit_instructions[key]):
                    if isinstance(self.fit_instructions[key][sfh_comp_key], tuple):
                        self.fit_limits.append(self.fit_instructions[key][sfh_comp_key])
                        self.fit_params.append(key + ":" + sfh_comp_key)
                        
                    elif isinstance(self.fit_instructions[key][sfh_comp_key], float) or isinstance(self.fit_instructions[key][sfh_comp_key], str):
                        if sfh_comp_key is not "type" and sfh_comp_key[-5:] != "prior" and self.fit_instructions[key][sfh_comp_key] != "hubble time":
                            self.fixed_values.append(self.fit_instructions[key][sfh_comp_key])
                            self.fixed_params.append(key + ":" + sfh_comp_key)

        # Populate list of priors
        for fit_param in self.fit_params:

            if len(fit_param.split(":")) == 1 and fit_param.split(":")[0] + "prior" in list(self.fit_instructions):
                    self.priors.append(self.fit_instructions[fit_param.split(":")[0] + "prior"])

            elif len(fit_param.split(":")) == 2 and fit_param.split(":")[1] + "prior" in list(self.fit_instructions[fit_param.split(":")[0]]):
                    self.priors.append(self.fit_instructions[fit_param.split(":")[0]][fit_param.split(":")[1] + "prior"])

            else:
                self.priors.append("uniform")

        """
        # Sets the max_zred parameter to just above the maximum fitted redshift in order to speed up model generation when fitting spectra
        if "zred" in self.fit_params:
            max_zred = self.fit_limits[self.fit_params.index("zred")][1] + 0.05

        elif "zred" in self.fixed_params:
            max_zred = self.fixed_values[self.fixed_params.index("zred")] + 0.05
        """
        self.ndim = len(self.fit_params)



    def fit(self, verbose=False, sampling_efficiency="model", n_live=400, sampler="pmn"):
        """ Fit the specified model to the input galaxy data using the MultiNest algorithm. 

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get extra progress updates from MultiNest.

        sampling_efficiency : str - optional
            MultiNest sampling efficiency parameter (see MultiNest papers), can be changed to "parameter" to speed things up, but may lead to unreliable results.

        n_live : int - optional
            Number of MultiNest live points. Reducing speeds up the code but may lead to unreliable results.

        """

        print("\nBagpipes: fitting object " + self.Galaxy.ID + "\n")
        start_time = time.time()


        if sampler == "pmn":
            pmn.run(self.get_lnprob, self.prior_transform, self.ndim, importance_nested_sampling = False, verbose = verbose, sampling_efficiency = sampling_efficiency, n_live_points = n_live, outputfiles_basename=working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-")

            a = pmn.Analyzer(n_params = self.ndim, outputfiles_basename=working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-", verbose=False)

            s = a.get_stats()

            self.global_log_evidence = s["nested sampling global log-evidence"]
            self.global_log_evidence_err = s["nested sampling global log-evidence error"]

            self.posterior["samples"] = np.loadtxt(working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-post_equal_weights.dat")[:,:-1]
            self.posterior["lnprob"] = np.loadtxt(working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-post_equal_weights.dat", usecols=(-1))


        elif sampler == "dynesty":

            if self.ndim <= 5:
                walks = 10

            elif self.ndim <= 10:
                walks = 25

            else:
                walks = 50

            if os.path.exists(working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-dynesty_post.dat"):
                self.posterior["samples"] = np.loadtxt(working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-dynesty_post.dat")[:,:-1]
                self.posterior["lnprob"] = np.loadtxt(working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-dynesty_post.dat", usecols=(-1))
                
                self.global_log_evidence, self.global_log_evidence_err = np.loadtxt(working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-dynesty_evidence.dat")

                print("Bagpipes: Previous dynesty run output detected, loading posterior.")

            else:
                self.sampler = dynesty.NestedSampler(self.get_lnprob, self.prior_transform, self.ndim, nlive=n_live, bound="multi", sample="rwalk", walks=walks)#, bootstrap=0, enlarge=1.25, vol_dec=0.8, vol_check=1.5, first_update={'min_ncall': -np.inf, 'min_eff': np.inf})
                self.sampler.run_nested(dlogz=0.01, print_progress=verbose)

                weights = np.exp(self.sampler.results.logwt - self.sampler.results.logz[-1])
                weights /= weights.sum()
                self.posterior["samples"] = dynesty.utils.resample_equal(self.sampler.results.samples, weights)
                self.posterior["lnprob"] = self.sampler.results.logl

                self.global_log_evidence = self.sampler.results.logz[-1]
                self.global_log_evidence_err = np.std(np.array([dynesty.utils.simulate_run(self.sampler.results).logz[-1] for i in range(50)]))

                output = np.zeros((self.posterior["samples"].shape[0], self.posterior["samples"].shape[1]+1))
                output[:,:-1] = self.posterior["samples"]
                output[:,-1] = self.posterior["lnprob"]

                np.savetxt(working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-dynesty_evidence.dat", np.array([self.global_log_evidence, self.global_log_evidence_err]))
                np.savetxt(working_dir + "/pipes/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-dynesty_post.dat", output)


        for i in range(len(self.fit_params)):
            self.posterior[self.fit_params[i]] = self.posterior["samples"][:,i]

        self.posterior_median = np.zeros(self.ndim)
        for j in range(self.ndim):
            self.posterior_median[j] = np.median(self.posterior["samples"][:,j])

        self.conf_int = []
        for j in range(self.ndim):
            self.conf_int.append((np.percentile(self.posterior["samples"][:,j], 16), np.percentile(self.posterior["samples"][:,j], 84)))

        self.best_fit_params = self.posterior["samples"][np.argmax(self.posterior["lnprob"]),:]
        self.get_lnprob(self.best_fit_params, self.ndim, self.ndim)

        self.min_chisq = 0.
        self.min_chisq_red = 0.
        self.ndof = -self.ndim

        if self.Galaxy.spectrum_exists == True:
            self.min_chisq += self.chisq_spec
            self.ndof += self.Galaxy.spectrum.shape[0]

        if self.Galaxy.photometry_exists == True:
            self.min_chisq += self.chisq_phot
            self.ndof += self.Galaxy.photometry.shape[0]

        self.min_chisq_red = self.min_chisq/float(self.ndof)

        print("\nBagpipes: fitting complete in " + str("%.1f" % (time.time() - start_time)) + " seconds, confidence interval:")
        for x in range(self.ndim):
            print(str(np.round(self.conf_int[x], 4)), np.round(self.posterior_median[x], 4), self.fit_params[x])
        print("\n")

        self.get_model(self.posterior_median)
        self.get_post_info()



    def prior_transform(self, cube, ndim=0, nparam=0): 
        # Prior function for MultiNest algorithm.

        for i in range(self.ndim):

            if self.priors[i] == "uniform":
                cube[i] = self.fit_limits[i][0] + (self.fit_limits[i][1] - self.fit_limits[i][0])*cube[i]

            elif self.priors[i] == "log_10":
                cube[i] =  10**((np.log10(self.fit_limits[i][1]/self.fit_limits[i][0]))*cube[i] + np.log10(self.fit_limits[i][0]))

            elif self.priors[i] == "log_e":
                cube[i] =  np.exp((np.log(self.fit_limits[i][1]/self.fit_limits[i][0]))*cube[i] + np.log(self.fit_limits[i][0]))

            elif self.priors[i] == "pow_10":
                cube[i] =  np.log10((10**self.fit_limits[i][1] - 10**self.fit_limits[i][0])*cube[i] + 10**self.fit_limits[i][0])

            elif self.priors[i] == "1/x":
                cube[i] =  1./((1./self.fit_limits[i][1] - 1./self.fit_limits[i][0])*cube[i] + 1./self.fit_limits[i][0])

            elif self.priors[i] == "1/x^2":
                cube[i] =  1./np.sqrt((1./self.fit_limits[i][1]**2 - 1./self.fit_limits[i][0]**2)*cube[i] + 1./self.fit_limits[i][0]**2)

            elif self.priors[i] == "Gaussian":

                if len(self.fit_params[i].split(":")) == 1:
                    mu = self.fit_instructions[self.fit_params[i] + "priormu"]
                    sigma = self.fit_instructions[self.fit_params[i] + "priorsigma"]

                elif  len(self.fit_params[i].split(":")) == 2:
                    mu = self.fit_instructions[self.fit_params[i].split(":")[0]][self.fit_params[i].split(":")[1] + "priormu"]
                    sigma = self.fit_instructions[self.fit_params[i].split(":")[0]][self.fit_params[i].split(":")[1] + "priorsigma"]

                uniform_max = erf((self.fit_limits[i][1] - mu)/np.sqrt(2)/sigma)
                uniform_min = erf((self.fit_limits[i][0] - mu)/np.sqrt(2)/sigma)
                cube[i] = (uniform_max-uniform_min)*cube[i] + uniform_min
                cube[i] = sigma*np.sqrt(2)*erfinv(cube[i]) + mu

        return cube



    def get_lnprob(self, x, ndim=0, nparam=0):
        # Returns the log-probability for a given model sfh and parameter vector x. 

        self.get_model(x)

        # If the age of any model component is greater than the age of the Universe, return a huge negative value for lnprob.
        if self.Model.sfh.maxage <= self.Model.sfh.age_of_universe:
            if "hypspec" in list(self.model_components):
                self.hyp_spec = self.model_components["hypspec"]

            if "hypphot" in list(self.model_components):
                self.hyp_phot = self.model_components["hypphot"]

            if self.Galaxy.spectrum_exists == True:
                self.chisq_spec = np.sum((self.Galaxy.spectrum[:,1] - self.Model.spectrum[:,1])**2/self.Galaxy.spectrum[:,2]**2)

            if self.Galaxy.photometry_exists == True:
                self.chisq_phot = np.sum((self.Galaxy.photometry[:,1] - self.Model.photometry)**2/self.Galaxy.photometry[:,2]**2)

            """
            if self.Galaxy.no_of_spectra > 1:
                for i in range(self.Galaxy.no_of_spectra - 1):
                    K_spec += -0.5*np.sum(np.log(2*np.pi*self.Galaxy.extra_spectra[i][:,2]**2))
                    chisq_spec = np.sum((self.Galaxy.extra_spectra[i][:,1] - self.extra_models[i].spectrum[:,1])**2/(self.Galaxy.extra_spectra[i][:,2]**2))
            """

            lnprob = self.K_phot + self.K_spec + 0.5*self.N_spec*np.log(self.hyp_spec) + 0.5*self.N_phot*np.log(self.hyp_phot) - 0.5*self.hyp_phot*self.chisq_phot - 0.5*self.hyp_spec*self.chisq_spec 

        else:
            lnprob = -9.99*10**99

        return lnprob



    def get_model(self, param):
        # Generates a model object for the a specified set of parameters.

        self.model_components = self.get_model_components(param)
        """
        if self.Galaxy.no_of_spectra > 1:

            del self.model_components["veldisp"]
            del self.model_components["polynomial"]

            self.model_components["veldisp"] = self.model_components["veldisp1"]
            self.model_components["polynomial"] = self.model_components["polynomial1"]
        """
        if self.Model is None:
            if self.Galaxy.spectrum_exists == True:
                self.Model = Model_Galaxy(self.model_components, self.Galaxy.filtlist, spec_wavs=self.Galaxy.spectrum[:,0])

            else:
                self.Model = Model_Galaxy(self.model_components, self.Galaxy.filtlist)

        else:
            self.Model.update(self.model_components)
        """
        if self.Galaxy.no_of_spectra > 1:

            for i in range(self.Galaxy.no_of_spectra-1):

                self.model_components["veldisp"] = self.model_components["veldisp" + str(i+2)]
                if "polynomial" + str(i+2) in list(self.model_components):
                    self.model_components["polynomial"] = self.model_components["polynomial" + str(i+2)]

                else:
                    del self.model_components["polynomial"]

                if self.extra_models[i] is None:
                    self.extra_models[i] = Model_Galaxy(self.model_components, spec_wavs=self.Galaxy.extra_spectra[i][:,0])
                
                else:
                    self.extra_models[i].update(self.model_components)
        """


    def get_model_components(self, param):
        # Turns a vector of parameters into a model_components dict, if input is already a dict simply returns it. 

        # If param is a model_components dictionary get right on with calculating the chi squared value
        if isinstance(param, dict):
            model_components = param

        # Otherwise assume it is a vector amd generate a model_components dict
        else:
            model_components = deepcopy(self.fit_instructions)

            # inserts the values of the fit parameters into model_components from the parameter vector x passed to the function
            for i in range(len(self.fit_params)):  
                if len(self.fit_params[i].split(":")) == 1:
                    model_components[self.fit_params[i]] = param[i]
                    
                elif  len(self.fit_params[i].split(":")) == 2:
                    model_components[self.fit_params[i].split(":")[0]][self.fit_params[i].split(":")[1]] = param[i]

            #finds any dependent parameters, which are set to the values of the parameters they depend on
            for i in range(len(self.fixed_values)): 
                if isinstance(self.fixed_values[i], str):
                    model_components[self.fixed_params[i].split(":")[0]][self.fixed_params[i].split(":")[1]] = model_components[self.fixed_values[i].split(":")[0]][self.fixed_values[i].split(":")[1]]

        return model_components



    def get_post_info(self):
        # Calculates a whole bunch of useful posterior quantities from the MultiNest output. 

        if "sfh" not in list(self.posterior):

            nsamples = self.posterior["samples"].shape[0]

            self.posterior["sfh"] = np.zeros((self.Model.sfh.ages.shape[0], nsamples))
            self.posterior["sfr"] = np.zeros(nsamples)
            self.posterior["mwa"] = np.zeros(nsamples)
            self.posterior["tmw"] = np.zeros(nsamples)
            self.posterior["UVJ"] = np.zeros((3, nsamples))

            self.posterior["living_stellar_mass"] = {}
            self.posterior["living_stellar_mass"]["total"] = np.zeros(nsamples)

            for comp in self.Model.sfh_components:
                self.posterior["living_stellar_mass"][comp] = np.zeros(nsamples)

            if self.Galaxy.photometry_exists == True:
                self.posterior["photometry"] = np.zeros((self.Model.photometry.shape[0], nsamples))

            if self.Galaxy.spectrum_exists == True:
                self.posterior["spectrum"] = np.zeros((self.Model.spectrum.shape[0], nsamples))
                self.posterior["polynomial"] = np.zeros((self.Model.spectrum.shape[0], nsamples)) + 1.
            """
            if self.Galaxy.no_of_spectra > 1:
                self.posterior["extra_spectra"] = []
                self.posterior["extra_polynomials"] = []

                for i in range(self.Galaxy.no_of_spectra-1):
                    self.posterior["extra_spectra"].append(np.zeros((self.extra_models[i].spectrum.shape[0], nsamples)))
                    self.posterior["extra_polynomials"].append(np.zeros((self.extra_models[i].spectrum.shape[0], nsamples)))
            """
            self.posterior["spectrum_full"] = np.zeros((self.Model.spectrum_full.shape[0], nsamples))

            for i in range(nsamples):
                self.get_model(self.posterior["samples"][i,:])

                self.posterior["sfh"][:,i] = self.Model.sfh.sfr 
                self.posterior["sfr"][i] = self.posterior["sfh"][0,i]
                self.posterior["mwa"][i] = (10**-9)*np.sum(self.Model.sfh.sfr*self.Model.sfh.ages*self.Model.sfh.age_widths)/np.sum(self.Model.sfh.sfr*self.Model.sfh.age_widths)
                self.posterior["tmw"][i] = np.interp(self.model_components["redshift"], z_array, age_at_z) - self.posterior["mwa"][i]
                self.posterior["living_stellar_mass"]["total"][i] = self.Model.living_stellar_mass["total"]

                for comp in self.Model.sfh_components:
                    self.posterior["living_stellar_mass"][comp][i] = self.Model.living_stellar_mass[comp]

                if self.Model.filtlist is not None:
                    self.posterior["UVJ"][:,i] = self.Model.get_restframe_UVJ()

                if self.Model.filtlist is not None:
                    self.posterior["photometry"][:,i] = self.Model.photometry

                self.posterior["spectrum_full"][:,i] = self.Model.spectrum_full

                if self.Model.spec_wavs is not None:
                    if self.Model.polynomial is None:
                        self.posterior["spectrum"][:,i] = self.Model.spectrum[:,1]

                    else:
                        self.posterior["spectrum"][:,i] = self.Model.spectrum[:,1]/self.Model.polynomial
                        self.posterior["polynomial"][:,i] = self.Model.polynomial
                """
                if self.Galaxy.no_of_spectra > 1:
                    for j in range(self.Galaxy.no_of_spectra-1):
                        if self.extra_models[j].polynomial is None:
                            self.posterior["extra_spectra"][j][:,i] = self.extra_models[j].spectrum[:,1]

                        else:
                            self.posterior["extra_spectra"][j][:,i] = self.extra_models[j].spectrum[:,1]/self.extra_models[j].polynomial
                            self.posterior["extra_polynomials"][j][:,i] = self.extra_models[j].polynomial
                """


    def plot_fit(self, return_fig=False):
        """ Generate a plot of the input data and fitted posterior spectrum/photometry. 
    
        Parameters
        ----------

        return_fig : bool - optional
            If True, returns the figure containing the fit to the user instead of saving it to the pipes/plots/run/ directory.

        """

        normalisation_factor = 10**18

        self.get_post_info()

        # Set up plot with the correct number of axes.
        naxes = self.Galaxy.no_of_spectra

        if self.Galaxy.photometry_exists == True:
            naxes += 1

        fig, axes = plt.subplots(naxes, figsize=(12, 4.*naxes))

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)

        if naxes == 1:
            axes = [axes]

        ax1 = axes[0]
        ax2 = axes[-1]

        # Set axis labels
        if naxes == 1:
            ax1.set_xlabel("$\lambda\ \\Big(\mathrm{\AA}\\Big)$")
            ax2.set_xlabel("$\mathrm{log_{10}}\\Big(\lambda / \mathrm{\AA}\\Big)$")

        else:
            for i in range(naxes-1):
                axes[i].set_xlabel("$\lambda\ \\Big(\mathrm{\AA}\\Big)$")
            ax2.set_xlabel("$\mathrm{log_{10}}\\Big(\lambda / \mathrm{\AA}\\Big)$")

        if self.fit_instructions["redshift"] != 0.:
            ylabel = "$\mathrm{f_{\lambda}}\ \mathrm{/\ 10^{-18}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$"

        else:
            ylabel = "$\mathrm{f_{\lambda}}\ \mathrm{/\ erg\ s^{-1}\ \AA^{-1}}$"

        if naxes > 1:
            fig.text(0.08, 0.63, ylabel, rotation=90)

        else:
            ax1.set_ylabel(ylabel, rotation=90)


        # Plot first spectrum
        if self.Galaxy.spectrum_exists == True:
            ax1.set_xlim(self.Galaxy.spectrum[0,0], self.Galaxy.spectrum[-1,0])

            if "polynomial" in list(self.model_components) or "polynomial1" in list(self.model_components):
                polynomial = np.median(self.posterior["polynomial"], axis=1)

            else:
                polynomial = np.ones(self.Galaxy.spectrum[:,0].shape[0])

            #ax1.plot(self.Galaxy.spectrum[:, 0], normalisation_factor*self.Galaxy.spectrum[:, 1], color="red", zorder=10)

            ax1.plot(self.Galaxy.spectrum[:, 0], normalisation_factor*self.Galaxy.spectrum[:, 1]/polynomial, color="dodgerblue", zorder=1, lw=2)
            #ax1.fill_between(self.Galaxy.spectrum[:, 0], normalisation_factor*(self.Galaxy.spectrum[:, 1]/polynomial - self.Galaxy.spectrum[:, 2]), normalisation_factor*(self.Galaxy.spectrum[:, 1]/polynomial + self.Galaxy.spectrum[:, 2]), color="dodgerblue", zorder=1, alpha=0.75, linewidth=0)

            ax1.set_ylim(0, 1.1*normalisation_factor*np.max(self.Galaxy.spectrum[:, 1]/polynomial))
        """
        # Plot any extra spectra
        if self.Galaxy.no_of_spectra > 1:
            for i in range(self.Galaxy.no_of_spectra-1):
                ax1.set_xlim(self.Galaxy.extra_spectra[i][0,0], self.Galaxy.extra_spectra[i][-1,0])
                polynomial = np.ones(self.Galaxy.extra_spectra[i][:, 0].shape[0])

            if "polynomial" + str(i+2) in list(self.model_components) or "polynomial1" in list(self.model_components):
                polynomial = np.median(self.posterior["extra_polynomials"][i+2], axis=1)

            else:
                polynomial = np.ones(self.Galaxy.spectrum[:,0].shape[0])

                axes[i+1].plot(self.Galaxy.extra_spectra[i][:, 0], self.Galaxy.extra_spectra[i][:, 1]/polynomial, color="dodgerblue", zorder=1)
                #axes[i+1].fill_between(self.Galaxy.spectrum[:, 0], self.Galaxy.extra_spectra[i][:, 1]/polynomial - self.Galaxy.extra_spectra[i][:, 2], self.Galaxy.extra_spectra[i][:, 1]/polynomial + self.Galaxy.extra_spectra[i][:, 2], color="dodgerblue", zorder=1, alpha=0.75, linewidth=0)
        """

        # Plot photometric data
        if self.Galaxy.photometry_exists == True:
            ax2.set_ylim(0., 1.1*np.max(normalisation_factor*self.Galaxy.photometry[:,1]))
            ax2.set_xlim((np.log10(self.Galaxy.photometry[0,0])-0.025), (np.log10(self.Galaxy.photometry[-1,0])+0.025))

            for axis in axes:
                axis.errorbar(np.log10(self.Galaxy.photometry[:,0][self.Galaxy.photometry[:,1] != 0.]), normalisation_factor*self.Galaxy.photometry[:,1][self.Galaxy.photometry[:,1] != 0.], yerr=normalisation_factor*self.Galaxy.photometry[:,2][self.Galaxy.photometry[:,1] != 0.], lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=3, color="black")
                axis.scatter(np.log10(self.Galaxy.photometry[:,0][self.Galaxy.photometry[:,1] != 0.]), normalisation_factor*self.Galaxy.photometry[:,1][self.Galaxy.photometry[:,1] != 0.], color="blue", s=50, zorder=4, linewidth=1, facecolor="blue", edgecolor="black")


        # Add masked regions to plots
        if os.path.exists(working_dir + "/pipes/object_masks/" + self.Galaxy.ID + "_mask") and self.Galaxy.spectrum_exists:
            mask = np.loadtxt(working_dir + "/pipes/object_masks/" + self.Galaxy.ID + "_mask")

            for j in range(self.Galaxy.no_of_spectra):
                if len(mask.shape) == 1:
                    axes[j].axvspan(mask[0], mask[1], color="gray", alpha=0.8, zorder=4)

                if len(mask.shape) == 2:
                    for i in range(mask.shape[0]):
                        axes[j].axvspan(mask[i,0], mask[i,1], color="gray", alpha=0.8, zorder=4)
        
        # Plot model posterior
        self.get_post_info()

        if self.Galaxy.photometry_exists == True:

            if "redshift" in list(self.posterior):
                z_plot = np.median(self.posterior["redshift"])

            else:
                z_plot = self.Model.model_comp["redshift"]

            ax2.fill_between(np.log10(self.Model.chosen_wavs*(1.+z_plot)), normalisation_factor*np.percentile(self.posterior["spectrum_full"], 16, axis=1), normalisation_factor*np.percentile(self.posterior["spectrum_full"], 84, axis=1), color="navajowhite", zorder=1, linewidth=0)
            ax2.plot(np.log10(self.Model.chosen_wavs*(1.+z_plot)), normalisation_factor*np.percentile(self.posterior["spectrum_full"], 16, axis=1), color="navajowhite", zorder=1)
            ax2.plot(np.log10(self.Model.chosen_wavs*(1.+z_plot)), normalisation_factor*np.percentile(self.posterior["spectrum_full"], 84, axis=1), color="navajowhite", zorder=1)

            ax2.set_ylim(0., np.max([ax2.get_ylim()[1], 1.1*np.max(normalisation_factor*np.percentile(self.posterior["spectrum_full"], 84, axis=1)[(self.Model.chosen_wavs*(1.+self.Model.model_comp["redshift"]) < 10**ax2.get_xlim()[1]) & (self.Model.chosen_wavs*(1.+self.Model.model_comp["redshift"]) > 10**ax2.get_xlim()[0])])]))

            for j in range(self.Model.photometry.shape[0]):
                phot_1sig = self.posterior["photometry"][j,(self.posterior["photometry"][j,:] > np.percentile(self.posterior["photometry"][j,:], 16)) & (self.posterior["photometry"][j,:] < np.percentile(self.posterior["photometry"][j,:], 84))]
                ax2.scatter(np.log10(np.zeros(phot_1sig.shape[0]) + self.Model.eff_wavs[j]), normalisation_factor*phot_1sig, color="darkorange", zorder=2, alpha=0.05, s=100, rasterized=True)
                
        if self.Galaxy.spectrum_exists == True:
            ax1.fill_between(self.Model.spectrum[:,0], normalisation_factor*np.percentile(self.posterior["spectrum"], 16, axis=1), normalisation_factor*np.percentile(self.posterior["spectrum"], 84, axis=1), color="sandybrown", zorder=2, alpha=0.75, linewidth=0)
            ax1.plot(self.Model.spectrum[:,0], normalisation_factor*np.percentile(self.posterior["spectrum"], 50, axis=1), color="sandybrown", zorder=2, lw=1.5)

            ax1.set_ylim(0, np.max([ax1.get_ylim()[1], 1.1*np.max(normalisation_factor*np.percentile(self.posterior["spectrum"], 84, axis=1))]))
            """
            if self.Galaxy.no_of_spectra > 1:
                for j in range(self.Galaxy.no_of_spectra-1):
                    axes[j+1].fill_between(self.extra_models[j].spectrum[:,0], normalisation_factor*np.percentile(self.posterior["extra_spectra"], 16, axis=1), np.percentile(self.posterior["extra_spectra"], 84, axis=1), color="sandybrown", zorder=2, alpha=0.5, linewidth=0)
                    axes[j+1].plot(self.extra_models[j].spectrum[:,0], normalisation_factor*np.percentile(self.posterior["extra_spectra"], 16, axis=1), color="sandybrown", zorder=2, alpha=0.5)
                    axes[j+1].plot(self.extra_models[j].spectrum[:,0], normalisation_factor*np.percentile(self.posterior["extra_spectra"], 84, axis=1), color="sandybrown", zorder=2, alpha=0.5)    
            """
        #axes[0].annotate("ID: " + str(self.Galaxy.ID), xy=(0.1*ax1.get_xlim()[1] + 0.9*ax1.get_xlim()[0], 0.95*ax1.get_ylim()[1] + 0.05*ax1.get_ylim()[0]), zorder=5)      

        if return_fig:
            return fig

        else:
            fig.savefig(working_dir + "/pipes/plots/" + self.run + "/" + self.Galaxy.ID + "_fit.pdf", bbox_inches="tight")
            plt.close(fig)



    def plot_poly(self, style="percentiles"):
        # Plot the posterior for the polynomial correction applied to the spectrum. 

        self.get_post_info()

        plt.figure()

        if style == "individual":
            for i in range(self.posterior["polynomial"].shape[1]):
                plt.plot(self.Model.spectrum[:,0], np.ones(self.Model.spectrum.shape[0], dtype=float)/self.posterior["polynomial"][:,i], color="gray", alpha=0.05)

        elif style == "percentiles":

            polyarray = np.ones_like(self.posterior["polynomial"]).astype(float)/self.posterior["polynomial"]

            plt.fill_between(self.Model.spectrum[:,0], np.percentile(polyarray, 16, axis=1), np.percentile(polyarray, 84, axis=1), color="navajowhite", alpha=0.75, zorder=10)
            plt.plot(self.Model.spectrum[:,0], np.percentile(polyarray, 16, axis=1), color="navajowhite", zorder=10)
            plt.plot(self.Model.spectrum[:,0], np.percentile(polyarray, 50, axis=1), color="darkorange", zorder=10)
            plt.plot(self.Model.spectrum[:,0], np.percentile(polyarray, 84, axis=1), color="navajowhite", zorder=10)

        plt.savefig(working_dir + "/pipes/plots/" + self.run + "/" + self.Galaxy.ID + "_polynomial.pdf")
        plt.close()



    def plot_corner(self, return_fig=False, truths=None, ranges=None, param_names_tolog=[]):
        """ Make a corner plot of the fitting parameters. 

        Parameters
        ----------

        return_fig : bool - optional
            If True, returns the figure containing the fit to the user instead of saving it to the pipes/plots/run/ directory.

        truths : list - optional
            List of true parameter values in the same order as Fit.fit_params, if supplied adds the true values to the corner plot.

        ranges : list - optional
            List of tuples containing upper and lower limits for the plot areas for each parameter (same order as Fit.fit_params).

        param_names_tolog : list - optional
            List of parameter names formatted as in Fit.fit_params. The log_10 of the posteriors for these parameters will be plotted.

        """

        param_cols_toplot = []
        param_names_toplot = []
        param_truths_toplot = []
        params_tolog = []

        plot_range = []

        for i in range(len(self.fit_params)):
            param_cols_toplot.append(i)
            param_names_toplot.append(self.fit_params[i])
            param_truths_toplot.append(self.posterior_median[i])
            
            plot_range.append(self.fit_limits[i])
            
            if self.fit_params[i] in param_names_tolog:
                params_tolog.append(len(param_cols_toplot)-1)
                plot_range[-1] = (np.log10(plot_range[-1][0]), np.log10(plot_range[-1][1]))

        # log parameters passed to the function in param_names_tolog
        for i in range(len(params_tolog)):
            self.posterior["samples"][:,params_tolog[i]] = np.log10(self.posterior["samples"][:,params_tolog[i]])

        reference_param_names = ["dblplaw:tau", "dblplaw:alpha", "dblplaw:beta", "dblplaw:metallicity", "dblplaw:massformed", "dust:Av", "redshift"]
        latex_param_names = ["$\\tau\ /\ \mathrm{Gyr}$", "$\mathrm{log_{10}}(\\alpha)$", "$\mathrm{log_{10}}(\\beta)$", "$Z\ /\ \mathrm{Z_\odot}$", "$\mathrm{log_{10}}\\big(\\frac{M_\mathrm{formed}}{\mathrm{M_\odot}}\\big)$", "$A_V$", "$z$"]

        for i in range(len(param_names_toplot)):
            if param_names_toplot[i] in reference_param_names:
                param_names_toplot[i] = latex_param_names[reference_param_names.index(param_names_toplot[i])]

        fig = corner.corner(self.posterior["samples"][:,param_cols_toplot], labels=param_names_toplot, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 14}, smooth=1.5, smooth1d=0.5, truths=truths, range=ranges)
        
        sfh_ax = fig.add_axes([0.65, 0.59, 0.32, 0.15], zorder=10)
        sfr_ax = fig.add_axes([0.82, 0.82, 0.15, 0.15], zorder=10)
        tmw_ax = fig.add_axes([0.65, 0.82, 0.15, 0.15], zorder=10)

        # Generate posterior SFR quantities
        self.get_post_info()
        
        self.plot_sfh_post(sfh_ax, style="step")

        # Plot the current star formation rate posterior
        sfr_ax.hist(self.posterior["sfr"], bins=20, color="white", density=True, histtype="step", edgecolor="black", lw=1.5, range=(np.max([0., np.mean(self.posterior["sfr"]) - 3*np.std(self.posterior["sfr"])]), np.mean(self.posterior["sfr"]) + 3*np.std(self.posterior["sfr"])))
        sfr_ax.set_xlabel("$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")
        sfr_ax.set_xlim(np.max([0., np.mean(self.posterior["sfr"]) - 3*np.std(self.posterior["sfr"])]), np.mean(self.posterior["sfr"]) + 3*np.std(self.posterior["sfr"]))
        sfr_ax.set_yticklabels([])

        # Plot the mass weighted age posterior
        tmw_ax.hist(self.posterior["tmw"], bins=20, color="white", density=True, histtype="step", edgecolor="black", lw=1.5, range=(np.max([0., np.mean(self.posterior["tmw"]) - 3*np.std(self.posterior["tmw"])]), np.mean(self.posterior["tmw"]) + 3*np.std(self.posterior["tmw"])))
        tmw_ax.set_xlabel("$t(z_\mathrm{form})\ /\ \mathrm{Gyr}$")
        tmw_ax.set_xlim(np.mean(self.posterior["tmw"]) - 3*np.std(self.posterior["tmw"]), np.mean(self.posterior["tmw"]) + 3*np.std(self.posterior["tmw"]))
        tmw_ax.set_yticklabels([])

        sfr_ax.axvline(np.percentile(self.posterior["sfr"], 16), linestyle="--", color="black")
        sfr_ax.axvline(np.percentile(self.posterior["sfr"], 50), linestyle="--", color="black")
        sfr_ax.axvline(np.percentile(self.posterior["sfr"], 84), linestyle="--", color="black")
        #sfr_ax.axvline(self.sfr_maxprob, color="#4682b4")

        tmw_ax.axvline(np.percentile(self.posterior["tmw"], 16), linestyle="--", color="black")
        tmw_ax.axvline(np.percentile(self.posterior["tmw"], 50), linestyle="--", color="black")
        tmw_ax.axvline(np.percentile(self.posterior["tmw"], 84), linestyle="--", color="black")
        #mwa_ax.axvline(self.mwa_maxprob, color="#4682b4")

        fig.text(0.725, 0.978, "$t(z_\mathrm{form})\ /\ \mathrm{Gyr} =\ " + str(np.round(np.percentile(self.posterior["tmw"], 50), 2)) + "^{+" + str(np.round(np.percentile(self.posterior["tmw"], 84) - np.percentile(self.posterior["tmw"], 50), 2)) + "}_{-" + str(np.round(np.percentile(self.posterior["tmw"], 50) - np.percentile(self.posterior["tmw"], 16), 2)) + "}$", horizontalalignment = "center", size=14)
        fig.text(0.895, 0.978, "$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}\ =\ " + str(np.round(np.percentile(self.posterior["sfr"], 50), 2)) + "^{+" + str(np.round(np.percentile(self.posterior["sfr"], 84) - np.percentile(self.posterior["sfr"], 50), 2)) + "}_{-" + str(np.round(np.percentile(self.posterior["sfr"], 50) - np.percentile(self.posterior["sfr"], 16), 2)) + "}$", horizontalalignment = "center", size=14)

        fig.savefig(working_dir + "/pipes/plots/" + self.run + "/" + self.Galaxy.ID + "_corner.pdf")

        if return_fig:
            return fig

        else:
            plt.close(fig)



    def plot_sfh_post(self, sfh_ax, style="smooth", colorscheme="bw", variable="sfr"):

        color1 = "black"
        color2 = "gray"

        if colorscheme == "irnbru":
            color1 = "darkorange"
            color2 = "navajowhite"

        if variable == "sfr":
            postgrid = self.posterior["sfh"]

        elif variable == "ssfr":
            self.posterior["ssfr"] = np.zeros_like(self.posterior["sfh"])

            for i in range(self.posterior["sfh"].shape[1]):

                for j in range(chosen_ages.shape[0]):

                    if np.sum(self.posterior["sfh"][j:,i]) != 0.:
                        self.posterior["ssfr"][j,i] = np.log10(self.posterior["sfh"][j,i]/np.sum(self.posterior["sfh"][j:,i]*chosen_age_widths[j:]))
                        #print(self.posterior["ssfr"][j,i])
                        #raw_input()
                    else:
                        self.posterior["ssfr"][j,i] = 0.

            postgrid = self.posterior["ssfr"]

        if style == "step":
            # Generate and populate sfh arrays which allow the SFH to be plotted with straight lines across bins of SFH
            sfh_x = np.zeros(2*self.Model.sfh.ages.shape[0])
            sfh_y = np.zeros(2*self.Model.sfh.sfr.shape[0])
            sfh_y_low = np.zeros(2*self.Model.sfh.sfr.shape[0])
            sfh_y_high = np.zeros(2*self.Model.sfh.sfr.shape[0])

            for j in range(self.Model.sfh.sfr.shape[0]):

                sfh_x[2*j] = self.Model.sfh.age_lhs[j]

                sfh_y[2*j] = np.median(postgrid[j,:])
                sfh_y[2*j + 1] = sfh_y[2*j]

                sfh_y_low[2*j] = np.percentile(postgrid[j,:], 16)
                sfh_y_low[2*j + 1] = sfh_y_low[2*j]

                if variable == "sfr":

                    if sfh_y_low[2*j] < 0:
                        sfh_y_low[2*j] = 0.

                    if sfh_y_low[2*j+1] < 0:
                        sfh_y_low[2*j+1] = 0.

                sfh_y_high[2*j] = np.percentile(postgrid[j,:], 84)
                sfh_y_high[2*j + 1] = sfh_y_high[2*j]

                if j == self.Model.sfh.sfr.shape[0]-1:
                    sfh_x[-1] = self.Model.sfh.age_lhs[-1] + 2*(self.Model.sfh.ages[-1] - self.Model.sfh.age_lhs[-1])

                else:
                    sfh_x[2*j + 1] = self.Model.sfh.age_lhs[j+1]

            # Plot the SFH
            sfh_ax.fill_between(np.interp(self.model_components["redshift"], z_array, age_at_z) - sfh_x*10**-9, sfh_y_low, sfh_y_high, color=color2, alpha=0.75, zorder=10)
            sfh_ax.plot(np.interp(self.model_components["redshift"], z_array, age_at_z) - sfh_x*10**-9, sfh_y, color=color1, zorder=10)
            sfh_ax.plot(np.interp(self.model_components["redshift"], z_array, age_at_z) - sfh_x*10**-9, sfh_y_high, color=color2, zorder=10)
            sfh_ax.plot(np.interp(self.model_components["redshift"], z_array, age_at_z) - sfh_x*10**-9, sfh_y_low, color=color2, zorder=10)

        elif style == "smooth":
            sfh_ax.fill_between(np.interp(self.model_components["redshift"], z_array, age_at_z) - self.Model.sfh.ages*10**-9, np.percentile(postgrid, 16, axis=1), np.percentile(postgrid, 84, axis=1), color=color2)
            sfh_ax.plot(np.interp(self.model_components["redshift"], z_array, age_at_z) - self.Model.sfh.ages*10**-9, np.percentile(postgrid, 50, axis=1), color=color1)

        if variable == "sfr":
            sfh_ax.set_ylim(0., np.max([sfh_ax.get_ylim()[1], 1.1*np.max(sfh_y_high)]))
            sfh_ax.set_ylabel("$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")

        elif variable == "ssfr":
            sfh_ax.set_ylim(-12.5, -7.5)
            sfh_ax.set_ylabel("$\mathrm{sSFR\ /\ yr^{-1}}$")

        sfh_ax.set_xlim(np.interp(self.model_components["redshift"], z_array, age_at_z), 0)

        sfh_ax2 = sfh_ax.twiny()
        sfh_ax2.set_xticks(np.interp([0, 0.5, 1, 2, 4, 10], z_array, age_at_z))
        sfh_ax2.set_xticklabels(["$0$", "$0.5$", "$1$", "$2$", "$4$", "$10$"])
        sfh_ax2.set_xlim(sfh_ax.get_xlim())
        sfh_ax2.set_xlabel("$\mathrm{Redshift}$")

        sfh_ax.set_xlabel("$\mathrm{Age\ of\ Universe\ (Gyr)}$")

        




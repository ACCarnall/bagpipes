import numpy as np
import os
import sys
import corner
import matplotlib.pyplot as plt

try:
    import pymultinest as pmn

except:
    print "BAGPIPES: Pymultinest not installed, fitting will not be available."

from matplotlib import gridspec
from scipy.optimize import minimize
from copy import deepcopy, copy
from scipy.special import erf, erfinv

import setup
import load_models
import model_galaxy

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
z_array = np.arange(0., 10., 0.01)
age_at_z = cosmo.age(z_array).value

from numpy.polynomial.chebyshev import chebval as cheb

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)



class Fit:

    def __init__(self, Galaxy, fit_instructions, mode=None, run="."):

        self.times = []

        print "BAGPIPES: Using model set:", setup.model_type

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

        # conf_int: A list of tuples representing the 16th and 84th percentiles of the posterior distributions for each fit parameter
        self.conf_int = []

        # best_fitvals: an array of the best fit (max. likelihood) parameters found
        self.best_fitvals = None

        # extra_models: a list into which extra model items will be placed if the galaxy object has more thna one spectrum
        if self.Galaxy.no_of_spectra > 1:
            self.extra_models = []
            for i in range(self.Galaxy.no_of_spectra-1):
                self.extra_models.append(None)

        # median_uncertainty: the median of the uncertainties on the spectral data points. Used for modelling intrinsic scatter.
        if self.Galaxy.spectrum_exists == True:
            self.median_sigma = np.median(self.Galaxy.spectrum[:,2])

        #Set up directories to contain the outputs, if they don't already exist
        if self.run is not ".":
            if not os.path.exists(setup.install_dir + "/pmn_chains"):
                os.mkdir(setup.install_dir + "/pmn_chains")

            if not os.path.exists(setup.install_dir + "/pmn_chains/" + self.run):
                os.mkdir(setup.install_dir + "/pmn_chains/" + self.run)

            if not os.path.exists(setup.install_dir + "/plots"):
                os.mkdir(setup.install_dir + "/plots")

            if not os.path.exists(setup.install_dir + "/plots/" + self.run):
                os.mkdir(setup.install_dir + "/plots/" + self.run)

        #Calculate the number of degrees of freedom

        # ndof: The number of degrees of freedom in the fit
        self.ndof = -self.ndim

        if self.Galaxy.spectrum_exists == True:
            self.ndof += self.Galaxy.spectrum.shape[0]

        if self.Galaxy.no_of_spectra > 1:
            for i in range(self.Galaxy.no_of_spectra - 1):
                self.ndof += self.Galaxy.extra_spectra[i].shape[0]

        if self.Galaxy.photometry_exists == True:
            self.ndof += self.Galaxy.photometry.shape[0]

        if self.Galaxy.spectrum_exists == True:
            for i in range(len(self.Galaxy.spectrum)):
                if self.Galaxy.spectrum[i,2] > 10**50:
                    self.ndof -= 1
                    
        if self.Galaxy.photometry_exists == True:
            for i in range(len(self.Galaxy.photometry)):
                if self.Galaxy.photometry[i,2] > 10**50:
                    self.ndof -= 1



    """Sets up the class by generating relevant variables from the input fit_instructions dictionary."""
    def process_fit_instructions(self):
        
        for key in self.fit_instructions.keys():
            if isinstance(self.fit_instructions[key], tuple):
                self.fit_limits.append(self.fit_instructions[key])
                self.fit_params.append(key)
                
            elif isinstance(self.fit_instructions[key], float):
                self.fixed_values.append(self.fit_instructions[key])
                self.fixed_params.append(key)
                
            elif isinstance(self.fit_instructions[key], dict):
                for sfh_comp_key in self.fit_instructions[key].keys():
                    if isinstance(self.fit_instructions[key][sfh_comp_key], tuple):
                        self.fit_limits.append(self.fit_instructions[key][sfh_comp_key])
                        self.fit_params.append(key + ":" + sfh_comp_key)
                        
                    elif isinstance(self.fit_instructions[key][sfh_comp_key], float) or isinstance(self.fit_instructions[key][sfh_comp_key], str):
                        if sfh_comp_key is not "type" and sfh_comp_key[-5:] != "prior" and self.fit_instructions[key][sfh_comp_key] != "hubble time":
                            self.fixed_values.append(self.fit_instructions[key][sfh_comp_key])
                            self.fixed_params.append(key + ":" + sfh_comp_key)

        # Populate list of priors
        for fit_param in self.fit_params:

            if len(fit_param.split(":")) == 1:
                if fit_param.split(":")[0] + "prior" in self.fit_instructions.keys():
                    self.priors.append(self.fit_instructions[fit_param.split(":")[0] + "prior"])

                else:
                    print "BAGPIPES: Warning, no prior specified on " + fit_param + ", adopting a uniform prior."
                    self.priors.append("uniform")

            elif len(fit_param.split(":")) == 2:
                if fit_param.split(":")[1] + "prior" in self.fit_instructions[fit_param.split(":")[0]].keys():
                    self.priors.append(self.fit_instructions[fit_param.split(":")[0]][fit_param.split(":")[1] + "prior"])

                else:
                    print "BAGPIPES: Warning, no prior specified on " + fit_param + ", adopting a uniform prior."
                    self.priors.append("uniform")

        # Sets the max_zred parameter to just above the maximum fitted redshift in order to speed up model generation when fitting spectra
        if "zred" in self.fit_params:
            setup.max_zred = self.fit_limits[self.fit_params.index("zred")][1] + 0.05

        elif "zred" in self.fixed_params:
            setup.max_zred = self.fixed_values[self.fixed_params.index("zred")] + 0.05

        self.ndim = len(self.fit_params)




    """ Run the MultiNest sampling algorithm and update the confidence intervals and best parameter dictionaries accordingly."""
    def fit(self, simplex=True):

        pmn.run(self.get_lnprob, self.prior, self.ndim, importance_nested_sampling = False, verbose = setup.pmn_verbose, sampling_efficiency = setup.sampling_efficiency, n_live_points = setup.n_live_points, outputfiles_basename=setup.install_dir + "/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-")

        a = pmn.Analyzer(n_params = self.ndim, outputfiles_basename=setup.install_dir + "/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-")

        s = a.get_stats()

        mode_evidences = []
        for i in range(len(s["modes"])):
            mode_evidences.append(s["modes"][i]["local log-evidence"])

        # Best fit values taken to be the highest point in probability surface
        self.best_fitvals = s["modes"][np.argmax(mode_evidences)]["maximum"]

        # Median values of the marginalised posterior distributions
        self.posterior_median = np.zeros(self.ndim)
        for j in range(self.ndim):
            self.posterior_median[j] = s["marginals"][j]["median"]

        """
        for i in range(len(s["modes"])):

            bestpar_mode = dict(zip(self.fit_params, s["modes"][i]["maximum"]))
            print "Mode number:", i
            print "Best params: ", bestpar_mode
            print "Mode log evidence:", s["modes"][i]["local log-evidence"]
            print "Min chisq:", self.get_lnprob(s["modes"][i]["maximum"], self.ndim, self.ndim, mode="chisq")
            print "  "
        """
        
        # Best fit values determined by performing Nelder-Mead minimisation starting from the best fit output from MultiNest
        
        if simplex == True:

            print "BAGPIPES: Min Chisq Reduced: ", self.get_lnprob(self.best_fitvals, self.ndim, self.ndim, mode="chisq")/self.ndof

            print "BAGPIPES: Performing simplex minimisation."

            min_chi = []
            min_par = []

            minim = minimize(self.get_lnprob, self.best_fitvals, args=(self.ndim, self.ndim, "lnprob_neg"), method="Nelder-Mead")#, bounds=self.fit_limits)
            min_chi.append(minim["fun"])
            min_par.append(minim["x"])

            for i in range(9):
                minim = minimize(self.get_lnprob, self.best_fitvals+0.01*np.random.randn(self.ndim), args=(self.ndim, self.ndim, "lnprob_neg"), method="Nelder-Mead")#, bounds=self.fit_limits)
                min_chi.append(minim["fun"])
                min_par.append(minim["x"])

            self.best_fitvals = min_par[np.argmin(np.array(min_chi))]
            
        else:
            print "BAGPIPES: Skipping simplex minimisation step."

        self.median_param_dict = dict(zip(self.fit_params, self.posterior_median))
        self.best_param_dict = dict(zip(self.fit_params, self.best_fitvals))
        self.max_lnp = self.get_lnprob(self.best_fitvals, self.ndim, self.ndim)
        self.min_chisq_reduced = self.get_lnprob(self.best_fitvals, self.ndim, self.ndim, mode="chisq")/self.ndof

        self.global_evidence = s['nested sampling global log-evidence']

        for j in range(self.ndim):
            self.conf_int.append((s["marginals"][j]["1sigma"][0], s["marginals"][j]["1sigma"][1]))
        
        if setup.pmn_verbose == True:
            print "Min Chisq Reduced: ", self.min_chisq_reduced
            print " "
            print "Confidence interval:"
            for x in range(self.ndim):
                print str(np.round(self.conf_int[x], 4)), np.round(self.best_fitvals[x], 4), self.fit_params[x]
            print " "



    """ Prior function for MultiNest algorithm, currently just converts unit cube to uniform prior between set units. """
    def prior(self, cube, ndim, nparams): 
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



    """ Returns the log-probability for a given model sfh and parameter vector x. """
    def get_lnprob(self, x, ndim, nparam, mode="lnprob", verbose=False):

        self.nan_flag = False

        # if any of the parameters is outside of its fit limits, this returns a huge negative value for lnprob.
        if not isinstance(x, dict):
            for i in range(self.ndim):
                if not self.fit_limits[i][0] <= x[i] <= self.fit_limits[i][1]:
                    self.nan_flag = True

        chisq_spec, chisq_phot = 0., 0.
        K_phot, K_spec = 0., 0.

        # otherwise generate the model and calculate lnprob.
        if self.nan_flag is not True:

            self.get_model(x)

            # If the age of the model is greater than the age of the Universe, return a huge negative value for lnprob.
            if self.Model.sfh.maxage > np.interp(self.model_components["zred"], z_array, age_at_z):
                self.nan_flag = True

            if "intsig" in self.model_components.keys():
                intsig = self.model_components["intsig"]

            else:
                intsig = 0.

            if self.Galaxy.spectrum_exists == True and self.nan_flag == False:
                K_spec += -0.5*np.sum(np.log(2*np.pi*(self.Galaxy.spectrum[:,2]**2 + (self.median_sigma*intsig)**2)))
                chisq_spec += np.sum((self.Galaxy.spectrum[:,1] - self.Model.spectrum[:,1])**2/(self.Galaxy.spectrum[:,2]**2 + (self.median_sigma*intsig)**2))

            if self.Galaxy.photometry_exists == True and self.nan_flag == False:
                K_phot += -0.5*np.sum(np.log(2*np.pi*self.Galaxy.photometry[:,2]**2))
                chisq_phot += np.sum(((self.Galaxy.photometry[:,1] - self.Model.photometry)/self.Galaxy.photometry[:,2])**2)

            if self.Galaxy.no_of_spectra > 1:
                for i in range(self.Galaxy.no_of_spectra - 1):
                    K_spec += -0.5*np.sum(np.log(2*np.pi*self.Galaxy.extra_spectra[i][:,2]**2))
                    chisq_spec += np.sum((self.Galaxy.extra_spectra[i][:,1] - self.extra_models[i].spectrum[:,1])**2/(self.Galaxy.extra_spectra[i][:,2]**2))

        if self.nan_flag is True:
            chisq_spec, chisq_phot = 9.99*10**99, 9.99*10**99

        if mode == "lnprob" or mode == "lnprob_neg":
            lnprob =  K_phot + K_spec - 0.5*chisq_phot - 0.5*chisq_spec

            if mode == "lnprob":
                return lnprob

            if mode == "lnprob_neg":
                return -lnprob

        elif mode == "chisq":
            return chisq_phot + chisq_spec



    """ Generates a model object for the a specified set of parameters """
    def get_model(self, param):

        self.model_components = self.get_model_components(param)

        #if np.random.rand() < 0.001:
        #    print self.model_components

        if self.Galaxy.no_of_spectra > 1:

            self.model_components["veldisp"] = self.model_components["veldisp1"]
            self.model_components["polynomial"] = self.model_components["polynomial1"]

        if self.Model is None:
            if self.Galaxy.spectrum_exists == True:
                self.Model = model_galaxy.Model_Galaxy(self.model_components, self.Galaxy.field, output_specwavs=self.Galaxy.spectrum[:,0])

            else:
                self.Model = model_galaxy.Model_Galaxy(self.model_components, self.Galaxy.field)

        else:
            self.Model.update(self.model_components)

        if self.Galaxy.no_of_spectra > 1:

            for i in range(self.Galaxy.no_of_spectra-1):

                self.model_components["veldisp"] = self.model_components["veldisp" + str(i+2)]
                if "polynomial" + str(i+2) in self.model_components.keys():
                    self.model_components["polynomial"] = self.model_components["polynomial" + str(i+2)]

                else:
                    del self.model_components["polynomial"]

                if self.extra_models[i] is None:
                    self.extra_models[i] = model_galaxy.Model_Galaxy(self.model_components, output_specwavs=self.Galaxy.extra_spectra[i][:,0])
                
                else:
                    self.extra_models[i].update(self.model_components)



    # Turns a vector of parameters  into a model_components dict, if input is already a dict simply returns it
    def get_model_components(self, param):

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



    ### All of the below functions are for making pretty plots, they're not involved in the fitting process



    def get_post_sfh(self):

        self.posterior = np.loadtxt(setup.install_dir + "/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-post_equal_weights.dat")[:,:-1]

        nsamples = self.posterior.shape[0]

        sfh_arr = np.zeros((nsamples, self.Model.sfh.ages.shape[0]))
        self.sfr_post = np.zeros(nsamples)
        self.mwa_post = np.zeros(nsamples)
        self.mstar_liv_post = np.zeros(nsamples)

        for i in range(nsamples):
            self.get_model(self.posterior[int(np.random.rand()*self.posterior.shape[0]),:])

            sfh_arr[i,:] = self.Model.sfh.sfr 
            self.sfr_post[i] = sfh_arr[i,0]
            self.mwa_post[i] = (10**-9)*np.sum(self.Model.sfh.sfr*self.Model.sfh.ages*self.Model.sfh.age_widths)/np.sum(self.Model.sfh.sfr*self.Model.sfh.age_widths)
            self.mstar_liv_post[i] = self.Model.living_mstar

        self.mwa_post = np.interp(self.model_components["zred"], z_array, age_at_z) - self.mwa_post

        self.sfh_med = np.median(sfh_arr, axis=0)
        self.sfh_16 = np.percentile(sfh_arr, 16, axis=0)
        self.sfh_84 = np.percentile(sfh_arr, 84, axis=0)

        self.sfr_percentiles = (np.percentile(self.sfr_post, 16), np.percentile(self.sfr_post, 50), np.percentile(self.sfr_post, 84))
        self.mwa_percentiles = (np.percentile(self.mwa_post, 16), np.percentile(self.mwa_post, 50), np.percentile(self.mwa_post, 84))
        self.mstar_liv_percentiles = (np.percentile(self.mstar_liv_post, 16), np.percentile(self.mstar_liv_post, 50), np.percentile(self.mstar_liv_post, 84))

        self.get_model(param=self.best_fitvals)

        self.sfr_maxprob = self.Model.sfh.sfr[0]
        self.mwa_maxprob = (10**-9)*np.sum(self.Model.sfh.sfr*self.Model.sfh.ages*self.Model.sfh.age_widths)/np.sum(self.Model.sfh.sfr*self.Model.sfh.age_widths)
        self.mstar_liv_maxprob = self.Model.living_mstar

        self.get_model(param=self.best_fitvals)



    def plot_fit(self, mode="save"):

        if self.best_fitvals is None:
            sys.exit("BAGPIPES: Cannot plot the fit unless fitting has taken place!")

        naxes = self.Galaxy.no_of_spectra

        if self.Galaxy.photometry_exists == True:
            naxes += 1

        fig, axes = plt.subplots(naxes, figsize=(12, 4.*naxes))

        if naxes == 1:
            axes = [axes]

        ax1 = axes[0]
        ax2 = axes[-1]

        ax2.set_xlabel("$\mathrm{log_{10}}\\Big(\lambda / \mathrm{\AA}\\Big)$", size=16)

        if self.fit_instructions["zred"] != 0.:
            fig.text(0.08, 0.68, "$\mathrm{f_{\lambda}}$ $\mathrm{(erg\ s^{-1}\ cm^{-2}\ \AA^{-1})}$", size=16, rotation=90)

        else:
            fig.text(0.08, 0.68, "$\mathrm{f_{\lambda}}$ $\mathrm{(erg\ s^{-1}\ \AA^{-1})}$", size=16, rotation=90)

        # Plot spectral data
        if self.Galaxy.spectrum_exists == True:
            ax1.set_xlim(self.Galaxy.spectrum[0,0], self.Galaxy.spectrum[-1,0])
            polynomial = np.ones(self.Galaxy.spectrum[:,0].shape[0])

            if "polynomial" in self.model_components.keys():
                polynomial = self.get_polynomial("polynomial", self.Galaxy.spectrum[:,0])

            elif "polynomial1" in self.model_components.keys():
                polynomial = self.get_polynomial("polynomial1", self.Galaxy.spectrum[:,0])

            ax1.plot(self.Galaxy.spectrum[:, 0], self.Galaxy.spectrum[:, 1]/polynomial, color="dodgerblue", zorder=1)
            #ax1.fill_between(self.Galaxy.spectrum[:, 0], self.Galaxy.spectrum[:, 1]/polynomial - self.Galaxy.spectrum[:, 2], self.Galaxy.spectrum[:, 1]/polynomial + self.Galaxy.spectrum[:, 2], color="dodgerblue", zorder=1, alpha=0.75, linewidth=0)

        # Plot any extra spectra
        if self.Galaxy.no_of_spectra > 1:
            for i in range(self.Galaxy.no_of_spectra-1):
                polynomial = np.ones(self.Galaxy.extra_spectra[i][:, 0].shape[0])

                if "polynomial" + str(i+2) in self.model_components.keys():
                    polynomial = self.get_polynomial("polynomial" + str(i+2), self.Galaxy.extra_spectra[i][:, 1])

                axes[i+1].plot(self.Galaxy.extra_spectra[i][:, 0], self.Galaxy.extra_spectra[i][:, 1]/polynomial, color="dodgerblue", zorder=1)

        # Plot photometric data
        if self.Galaxy.photometry_exists == True:
            ax2.set_xlim((np.log10(self.Galaxy.photometry[0,0])-0.025), (np.log10(self.Galaxy.photometry[-1,0])+0.025))
            ax2.set_ylim(0., 1.1*np.max(self.Galaxy.photometry[:,1]))

            for axis in axes:
                axis.errorbar(np.log10(self.Galaxy.photometry[:,0]), self.Galaxy.photometry[:,1], yerr=self.Galaxy.photometry[:,2], lw=1.0, linestyle=" ", capsize=3, capthick=1, zorder=3, color="black")
                axis.scatter(np.log10(self.Galaxy.photometry[:,0]), self.Galaxy.photometry[:,1], color="blue", s=75, zorder=4, linewidth=1, facecolor="blue", edgecolor="black", label="Observed Photometry")

        # Add masked regions to plots
        if os.path.exists(setup.install_dir + "/object_masks/" + self.Galaxy.ID + "_mask") and self.Galaxy.spectrum_exists:
            mask = np.loadtxt(setup.install_dir + "/object_masks/" + self.Galaxy.ID + "_mask")

            for j in range(self.Galaxy.no_of_spectra):
                if len(mask.shape) == 1:
                    axes[j].axvspan(mask[0], mask[1], color="gray", alpha=0.8, zorder=4)

                if len(mask.shape) == 2:
                    for i in range(mask.shape[0]):
                        axes[j].axvspan(mask[i,0], mask[i,1], color="gray", alpha=0.8, zorder=4)
        
        # Generate model posterior
        self.posterior = np.loadtxt(setup.install_dir + "/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-post_equal_weights.dat")[:,:-1]

        nsamples = self.posterior.shape[0]
        
        if self.Galaxy.photometry_exists == True:
            post_phot = np.zeros((self.Model.photometry.shape[0], nsamples))

        if self.Galaxy.spectrum_exists == True:
            post_spec = np.zeros((self.Model.spectrum.shape[0], nsamples))

        if self.Galaxy.no_of_spectra > 1:
            extra_post_spec = []
            for i in range(self.Galaxy.no_of_spectra-1):
                extra_post_spec.append(np.zeros((self.extra_models[i].spectrum.shape[0], nsamples)))

        post_spec_full = np.zeros((self.Model.spectrum_full.shape[0], nsamples))

        for i in range(nsamples):
            self.get_model(self.posterior[int(np.random.rand()*self.posterior.shape[0]),:])

            if self.Galaxy.field is not None:
                post_phot[:,i] = self.Model.photometry

            post_spec_full[:,i] = self.Model.spectrum_full

            if self.Model.output_specwavs is not None:
                if self.Model.polynomial is None:
                    post_spec[:,i] = self.Model.spectrum[:,1]

                else:
                    post_spec[:,i] = self.Model.spectrum[:,1]/self.Model.polynomial

            if self.Galaxy.no_of_spectra > 1:
                for j in range(self.Galaxy.no_of_spectra-1):
                    if self.extra_models[j].polynomial is None:
                        extra_post_spec[j][:,i] = self.extra_models[j].spectrum[:,1]

                    else:
                        extra_post_spec[j][:,i] = self.extra_models[j].spectrum[:,1]/self.extra_models[j].polynomial
        
        # Plot model posterior
        if self.Galaxy.photometry_exists == True:
            ax2.fill_between(np.log10(self.Model.chosen_modelgrid_wavs*(1.+self.Model.model_comp["zred"])), np.percentile(post_spec_full, 16, axis=1), np.percentile(post_spec_full, 84, axis=1), color="navajowhite", zorder=1, linewidth=0)
            ax2.plot(np.log10(self.Model.chosen_modelgrid_wavs*(1.+self.Model.model_comp["zred"])), np.percentile(post_spec_full, 16, axis=1), color="navajowhite", zorder=1)
            ax2.plot(np.log10(self.Model.chosen_modelgrid_wavs*(1.+self.Model.model_comp["zred"])), np.percentile(post_spec_full, 84, axis=1), color="navajowhite", zorder=1)
            ax2.set_ylim(0., np.max([ax2.get_ylim()[1], 1.1*np.max(np.percentile(post_spec_full, 84, axis=1))]))


            for j in range(self.Model.photometry.shape[0]):
                ax2.scatter(np.log10(np.zeros(nsamples) + self.Model.phot_wavs[j]), post_phot[j,:], color="darkorange", zorder=2, alpha=0.05, s=150, rasterized=True)
                
        if self.Galaxy.spectrum_exists == True:
            ax1.fill_between(self.Model.spectrum[:,0], np.percentile(post_spec, 16, axis=1), np.percentile(post_spec, 84, axis=1), color="sandybrown", zorder=2, alpha=0.75, linewidth=0)
            ax1.plot(self.Model.spectrum[:,0], np.percentile(post_spec, 50, axis=1), color="sandybrown", zorder=2)

            if self.Galaxy.no_of_spectra > 1:
                for j in range(self.Galaxy.no_of_spectra-1):
                    axes[j+1].fill_between(self.extra_models[j].spectrum[:,0], np.percentile(extra_post_spec[j], 16, axis=1), np.percentile(extra_post_spec[j], 84, axis=1), color="sandybrown", zorder=2, alpha=0.5, linewidth=0)
                    axes[j+1].plot(self.extra_models[j].spectrum[:,0], np.percentile(extra_post_spec[j], 16, axis=1), color="sandybrown", zorder=2, alpha=0.5)
                    axes[j+1].plot(self.extra_models[j].spectrum[:,0], np.percentile(extra_post_spec[j], 84, axis=1), color="sandybrown", zorder=2, alpha=0.5)

        self.get_model(param=self.best_fitvals)
        

        #axes[0].annotate("ID: " + str(self.Galaxy.ID), xy=(0.1*ax1.get_xlim()[1] + 0.9*ax1.get_xlim()[0], 0.95*ax1.get_ylim()[1] + 0.05*ax1.get_ylim()[0]), size=12, zorder=5)      
   
        if mode == "show":
            plt.show()

        fig.savefig(setup.install_dir + "/plots/" + self.run + "/" + self.Galaxy.ID + "_fit.pdf", bbox_inches="tight")
        plt.close(fig)



    def plot_poly(self, mode="save"):

        if mode == "live":
            self.posterior = np.loadtxt(setup.install_dir + "/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-phys_live.points")[:,:-2]

        else:
            self.posterior = np.loadtxt(setup.install_dir + "/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-post_equal_weights.dat")[:,:-1]

        plt.figure()

        nsamples = 200

        if self.posterior.shape[0] < nsamples:
            nsamples = self.posterior.shape[0]

        for i in range(nsamples):
            self.get_model(self.posterior[int(np.random.rand()*self.posterior.shape[0]),:])
            plt.plot(self.Model.spectrum[:,0], np.ones(self.Model.spectrum.shape[0], dtype=float)/self.Model.polynomial, color="gray", alpha=0.05)

        plt.savefig(setup.install_dir + "/plots/" + self.run + "/" + self.Galaxy.ID + "_polynomial.pdf")
        plt.close()



    def plot_corner(self, print_param_values=False):

        if not os.path.exists(setup.install_dir + "/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-.txt"):
            print "BAGPIPES: Corner plot not made because previous run output file not found."
            return

        param_cols_toplot = []
        param_names_toplot = []
        param_truths_toplot = []
        params_tolog = []

        plot_range = []

        for i in range(len(self.fit_params)):
            if self.fit_params[i][:10] != "polynomial":
                param_cols_toplot.append(i)
                param_names_toplot.append(self.fit_params[i])
                param_truths_toplot.append(self.best_fitvals[i])
                
                plot_range.append(self.fit_limits[i])

                if self.priors[i] == "log_10" or self.priors[i] == "log_e":
                    #param_names_toplot[-1] = "log_" + param_names_toplot[-1]
                    #param_truths_toplot[-1] = np.log10(param_truths_toplot[-1])
                    params_tolog.append(len(param_cols_toplot)-1)
                    plot_range[i] = (np.log10(plot_range[i][0]), np.log10(plot_range[i][1]))
                
        self.posterior = np.loadtxt(setup.install_dir + "/pmn_chains/" + self.run + "/" + self.Galaxy.ID + "-post_equal_weights.dat", usecols=param_cols_toplot)

        # If a parameter has a logarithmic prior, plot the log of that parameter in the corner plot
        for i in range(len(params_tolog)):
            self.posterior[:,params_tolog[i]] = np.log10(self.posterior[:,params_tolog[i]])

        reference_param_names = ["dblplaw:tau", "dblplaw:alpha", "dblplaw:beta", "dblplaw:metallicity", "dblplaw:mass", "dust:Av"]
        latex_param_names = ["$\\tau\ (\mathrm{Gyr})$", "$\mathrm{log_{10}}(\\alpha)$", "$\mathrm{log_{10}}(\\beta)$", "$Z\ (Z_\odot )$", "$\mathrm{log_{10}}\\big(\\frac{M_\mathrm{formed}}{M_\odot}\\big)$", "$A_V$"]

        for i in range(len(param_names_toplot)):
            if param_names_toplot[i] in reference_param_names:
                param_names_toplot[i] = latex_param_names[reference_param_names.index(param_names_toplot[i])]

        ranges = [(0., 0.5), (4., np.interp(self.model_components["zred"], z_array, age_at_z)), (-2, 3), (0.2, 0.8), (10.15, 10.65), (0.5, 3)]

        fig = corner.corner(self.posterior, labels=param_names_toplot, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}, smooth="1.5", smooth1d="0.5")#, range=ranges)#truths=param_truths_toplot, 
        
        sfh_ax = fig.add_axes([0.65, 0.605, 0.32, 0.15], zorder=10)
        sfr_ax = fig.add_axes([0.82, 0.82, 0.15, 0.15], zorder=10)
        mwa_ax = fig.add_axes([0.65, 0.82, 0.15, 0.15], zorder=10)

        # Generate posterior SFR quantities
        self.get_post_sfh()

        # Generate and populate sfh arrays which allow the SFH to be plotted with straight lines across bins of SFH
        sfh_x = np.zeros(2*self.Model.sfh.ages.shape[0])
        sfh_y = np.zeros(2*self.Model.sfh.sfr.shape[0])
        sfh_y_low = np.zeros(2*self.Model.sfh.sfr.shape[0])
        sfh_y_high = np.zeros(2*self.Model.sfh.sfr.shape[0])

        for j in range(self.Model.sfh.sfr.shape[0]):

            sfh_x[2*j] = self.Model.sfh.age_lhs[j]

            sfh_y[2*j] = self.sfh_med[j]
            sfh_y[2*j + 1] = self.sfh_med[j]

            sfh_y_low[2*j] = self.sfh_16[j]
            sfh_y_low[2*j + 1] = self.sfh_16[j]

            if sfh_y_low[2*j] < 0:
                sfh_y_low[2*j] = 0.

            if sfh_y_low[2*j+1] < 0:
                sfh_y_low[2*j+1] = 0.

            sfh_y_high[2*j] = self.sfh_84[j]
            sfh_y_high[2*j + 1] = self.sfh_84[j]

            if j == self.Model.sfh.sfr.shape[0]-1:
                sfh_x[-1] = self.Model.sfh.age_lhs[-1] + 2*(self.Model.sfh.ages[-1] - self.Model.sfh.age_lhs[-1])
            else:
                sfh_x[2*j + 1] = self.Model.sfh.age_lhs[j+1]

        # Plot the SFH
        sfh_ax.fill_between(np.interp(self.model_components["zred"], z_array, age_at_z) - sfh_x*10**-9, sfh_y_low, sfh_y_high, color="gray", alpha=0.75)
        sfh_ax.plot(np.interp(self.model_components["zred"], z_array, age_at_z) - sfh_x*10**-9, sfh_y, color="black", zorder=10)
        sfh_ax.plot(np.interp(self.model_components["zred"], z_array, age_at_z) - sfh_x*10**-9, sfh_y_high, color="gray")
        sfh_ax.plot(np.interp(self.model_components["zred"], z_array, age_at_z) - sfh_x*10**-9, sfh_y_low, color="gray")
        sfh_ax.set_xlim(np.interp(self.model_components["zred"], z_array, age_at_z), 0)

        sfh_ax2 = sfh_ax.twiny()
        sfh_ax2.set_xticks(np.interp([0, 0.5, 1, 2, 4, 10], z_array, age_at_z))
        sfh_ax2.set_xticklabels(["$0$", "$0.5$", "$1$", "$2$", "$4$", "$10$"])
        sfh_ax2.set_xlim(sfh_ax.get_xlim())
        sfh_ax2.set_xlabel("$\mathrm{Redshift}$")

        # Plot the current star formation rate posterior
        sfr_ax.hist(self.sfr_post, bins=15, color="white", normed=True, histtype="step", edgecolor="black")
        sfr_ax.set_xlabel("$\mathrm{SFR\ (M_\odot\ yr^{-1}}$)")
        sfr_ax.set_xlim(0., np.mean(self.sfr_post) + 3*np.std(self.sfr_post))#np.mean(self.sfr_post) - 3*np.std(self.sfr_post)
        sfr_ax.set_yticklabels([])

        # Plot the mass weighted age posterior
        mwa_ax.hist(self.mwa_post, bins=15, color="white", normed=True, histtype="step", edgecolor="black")
        mwa_ax.set_xlabel("$t(z_\mathrm{form})\ (\mathrm{Gyr}$)")
        mwa_ax.set_xlim(np.mean(self.mwa_post) - 3*np.std(self.mwa_post), np.mean(self.mwa_post) + 3*np.std(self.mwa_post))
        mwa_ax.set_yticklabels([])

        sfr_ax.axvline(self.sfr_percentiles[0], linestyle="--", color="black")
        sfr_ax.axvline(self.sfr_percentiles[1], linestyle="--", color="black")
        sfr_ax.axvline(self.sfr_percentiles[2], linestyle="--", color="black")
        #sfr_ax.axvline(self.sfr_maxprob, color="#4682b4")

        sfh_ax.set_ylabel("$\mathrm{SFR}\ (M_\odot\ yr^{-1})$")
        sfh_ax.set_xlabel("$\mathrm{Age\ of\ Universe\ (Gyr)}$")
        sfh_ax.set_ylim(0, 1.1*np.max(sfh_y_high))

        mwa_ax.axvline(self.mwa_percentiles[0], linestyle="--", color="black")
        mwa_ax.axvline(self.mwa_percentiles[1], linestyle="--", color="black")
        mwa_ax.axvline(self.mwa_percentiles[2], linestyle="--", color="black")
        #mwa_ax.axvline(self.mwa_maxprob, color="#4682b4")


        fig.text(0.725, 0.978, "$t(z_\mathrm{form})\ (Gyr)\ =\ " + str(np.round(self.mwa_percentiles[1], 2)) + "^{+" + str(np.round(self.mwa_percentiles[2] - self.mwa_percentiles[1], 2)) + "}_{-" + str(np.round(self.mwa_percentiles[1] - self.mwa_percentiles[0], 2)) + "}$", size=12, horizontalalignment = "center")
        fig.text(0.895, 0.978, "$\mathrm{SFR}\ (M_\odot\ yr^{-1})\ =\ " + str(np.round(self.sfr_percentiles[1], 2)) + "^{+" + str(np.round(self.sfr_percentiles[2] - self.sfr_percentiles[1], 2)) + "}_{-" + str(np.round(self.sfr_percentiles[1] - self.sfr_percentiles[0], 2)) + "}$", size=12, horizontalalignment = "center")


        # Add parameter values to plots
        self.get_model(param=self.best_fitvals)

        if print_param_values == True:
            fig.text(0.78, 0.55, "Parameter", size=10, zorder=10, horizontalalignment="center", weight="bold")
            fig.text(0.87, 0.55, "Posterior", size=10, zorder=10, weight="bold", horizontalalignment="center")
            fig.text(0.94, 0.55, "Max. Prob.", size=10, zorder=10, weight="bold", horizontalalignment="center")

            line = 0
            
            for i in range(self.ndim):

                ndecimal = 0
                lowlim = np.round(self.posterior_median[i] - self.conf_int[i][0], ndecimal)
                highlim = np.round(self.conf_int[i][1] - self.posterior_median[i] , ndecimal)

                while lowlim == 0. or highlim == 0.:
                    lowlim = np.round(self.posterior_median[i]  - self.conf_int[i][0], ndecimal)
                    highlim = np.round(self.conf_int[i][1] - self.posterior_median[i] , ndecimal)
                    ndecimal += 1

                if ndecimal == 0:
                    ndecimal = 1


                lowlim = np.round(self.posterior_median[i]  - self.conf_int[i][0], ndecimal)
                highlim = np.round(self.conf_int[i][1] - self.posterior_median[i] , ndecimal)
                parval = str(np.round(self.posterior_median[i], ndecimal))
                maxp_value = np.round(self.best_fitvals[i], ndecimal)

                if self.fit_params[i][:10] != "polynomial":
                    fig.text(0.78, (0.53 - 0.0175*line), self.fit_params[i], horizontalalignment = "center", size=10, zorder=10)
                    fig.text(0.87, (0.53 - 0.0175*line), "$" + str(parval) + "^{+" + str(highlim) + "}_{-" + str(lowlim) + "}$", size=10, zorder=10, horizontalalignment = "center")
                    fig.text(0.94, (0.53 - 0.0175*line), maxp_value, size=10, zorder=10, horizontalalignment = "center")

                    line += 1.
            
            # Add chi-squared value to plot
            chisq_spec, chisq_phot = 0., 0.
            
            if "intsig" in self.model_components.keys():
                intsig = self.model_components["intsig"]

            else:
                intsig = 0.

            if self.Galaxy.spectrum_exists == True:
                chisq_spec += np.sum((self.Galaxy.spectrum[:,1] - self.Model.spectrum[:,1])**2/(self.Galaxy.spectrum[:,2]**2 + (self.median_sigma*intsig)**2))
     
            if self.Galaxy.photometry_exists == True:
                chisq_phot += np.sum(((self.Galaxy.photometry[:,1] - self.Model.photometry)/self.Galaxy.photometry[:,2])**2) 
                
            fig.text(0.44, 0.9725, "$\chi_\\nu^2\mathrm{ = " + str(np.round(self.min_chisq_reduced, 3)) + "\ frac\ spec=\ " + str(np.round(chisq_spec/self.min_chisq_reduced/self.ndof, 4)) + "\ frac\ phot=\ " + str(np.round(chisq_phot/self.min_chisq_reduced/self.ndof, 4)) + "}$", size=14, zorder=10, horizontalalignment = "center")      
            
            fig.text(0.44, 0.92, "ID: " + str(self.Galaxy.ID), size=18, zorder=10, horizontalalignment = "center")      

        fig.savefig(setup.install_dir + "/plots/" + self.run + "/" + self.Galaxy.ID + "_corner.pdf")

        plt.close(fig)



    def get_polynomial(self, polyname, x_vals):

        polycoefs = []
        while str(len(polycoefs)) in self.model_components[polyname].keys():
            polycoefs.append(self.model_components[polyname][str(len(polycoefs))])

        points = (x_vals - x_vals[0])/(x_vals[-1] - x_vals[0])

        return cheb(points, polycoefs)





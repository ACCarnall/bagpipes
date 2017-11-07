import numpy as np
import sys
import time
import matplotlib.pyplot as plt

sys.path.append("..")
import setup

import model_galaxy
import setup

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
z_array = np.arange(0., 10., 0.01)
age_at_z = cosmo.age(z_array).value

from scipy.optimize import fsolve


def lognorm_equations(p, consts):

    tau_solve, T0_solve = p

    xmax, h = consts

    return (np.exp(T0_solve - tau_solve**2) - xmax, xmax*(np.exp(0.5*np.sqrt(8*np.log(2)*tau_solve**2)) - np.exp(-0.5*np.sqrt(8*np.log(2)*tau_solve**2))) - h)


class Star_Formation_History:

    def __init__(self, model_components):

        self.ages = setup.chosen_ages
        self.age_lhs = setup.chosen_age_lhs[:-1]
        self.age_widths = setup.chosen_age_widths

        # component_types: List of all possible types of star formation history (SFH) components
        self.component_types = ["burst", "constant", "exponential", "delayed", "lognormal", "dblplaw", "lognormalgauss", "custom"]

        self.special_component_types = ["lognormal", "dblplaw", "lognormalgauss", "custom"]

        # model_components: dictionary containing all information about the model being generated    
        self.model_components = model_components

        # weight_widths: A dictionary which will contain arrays of SSP weights for different SFH components to allow CSPs to be made
        self.weight_widths = {}
        
        # populate the weight_widths dict with arrays of ssfr values by calling the functions corresponding to the SFH component type
        for component in self.model_components.keys():
            if component in self.component_types or component[:-1] in self.component_types:
                for option in self.component_types:
                    if component[:len(option)] == option:
                        self.weight_widths[component] = getattr(self, option)(self.model_components[component])

        # sfr: an array of total (mass normalised) star formation rate values across all SFH components 
        self.sfr = np.zeros(len(self.ages))

        # sum all of the weight_widths to get the total sfr(t)
        for component in self.model_components.keys():
            if component in self.component_types or component[:-1] in self.component_types:
                self.sfr += self.weight_widths[component]/self.age_widths

        # obtain the maximum age (time before observation) for which the ssfr is non-zero in Gyr (for plotting purposes)
        self.maxage = 0.

        if "lognormal" in self.model_components.keys() or "dblplaw" in self.model_components.keys() or "lognormalgauss" in self.model_components.keys() or "custom" in self.model_components.keys():
            self.maxage = np.interp(self.model_components["zred"], z_array, age_at_z)

        #Hacky prior on dblplaw:tau
        if "dblplaw" in self.model_components.keys():
            if self.model_components["dblplaw"]["tau"] > self.maxage:
                self.maxage = self.model_components["dblplaw"]["tau"]
        
        for component in self.model_components.keys():
            if component in self.component_types or component[:-1] in self.component_types:
                if component not in self.special_component_types and component[:-1] not in self.special_component_types:
                    if model_components[component]["age"] == "hubble time":
                        self.maxage = np.interp(self.model_components["zred"], z_array, age_at_z)
                                        
                    if model_components[component]["age"] > self.maxage:
                        self.maxage = model_components[component]["age"]




    """ returns an array of sfr(t) values for a burst component. """
    def burst(self, par_dict):

        # If a burst with finite width is requested, set up a constant component between the start and finish times.
        if "width" in par_dict.keys():
            if par_dict["age"] - par_dict["width"] < 0:
                sys.exit("BAGPIPES: The time at which the burst started was less than zero.")

            const_par_dict = {}
            const_par_dict["age"] = par_dict["age"]
            const_par_dict["age_min"] = par_dict["age"] - par_dict["width"]
            const_par_dict["mass"] = par_dict["mass"]

            weight_widths = self.constant(const_par_dict)
            return weight_widths

        else:

            widths_burst = np.copy(self.age_widths)
            weights_burst = np.zeros(len(widths_burst))
            
            age_ind = self.ages[self.ages < par_dict["age"]*10**9].shape[0]

            hiage_factor = ((par_dict["age"]*10**9 - self.ages[age_ind-1])/(self.ages[age_ind] - self.ages[age_ind-1]))
            lowage_factor = (1. - hiage_factor)

            weights_burst[age_ind-1] = lowage_factor
            weights_burst[age_ind] = hiage_factor

            weight_widths = weights_burst*widths_burst
            weight_widths /= np.sum(weight_widths)
            weight_widths *= 10**par_dict["mass"]

            return weight_widths



    """ returns an array of sfr(t) values for an exponential component normalised to one Solar mass of star formation by the present time. """
    def exponential(self, par_dict):
        
        age = par_dict["age"]*10**9
        if "tau" in par_dict.keys() and "nefolds" in par_dict.keys():
            sys.exit("BAGPIPES: For exponential component please specify one of tau or nefolds only.")

        if "nefolds" in par_dict.keys():    
            par_dict["tau"] = par_dict["age"]/par_dict["nefolds"]
        
        age_ind = self.ages[self.ages < par_dict["age"]*10**9].shape[0]

        widths_exponential = np.copy(self.age_widths)
        
        weights_exponential = np.zeros(widths_exponential.shape)
        weights_exponential[:age_ind+1] = np.exp(-(par_dict["age"]*10**9 - self.ages[:age_ind+1])/(par_dict["tau"]*10**9))

        if self.age_lhs[age_ind] - par_dict["age"]*10**9 > 0:
            widths_exponential[age_ind] = 0.
            widths_exponential[age_ind-1] = par_dict["age"]*10**9 - self.age_lhs[age_ind-1]
            
        else:
            widths_exponential[age_ind] = par_dict["age"]*10**9 - self.age_lhs[age_ind]

        weight_widths = weights_exponential*widths_exponential
        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["mass"]

        return weight_widths



    """ returns an array of sfr(t) values for a constant component normalised to one Solar mass of star formation by the present time. """
    def constant(self, par_dict):
        
        if par_dict["age"] == "hubble time":
            par_dict["age"] = np.interp(self.model_components["zred"], z_array, age_at_z)

        if par_dict["age_min"] > par_dict["age"]:
            sys.exit("Minimum constant age exceeded maximum.")
        
        age_ind = self.ages[self.ages < par_dict["age"]*10**9].shape[0]

        age_min_ind = self.ages[self.ages < par_dict["age_min"]*10**9].shape[0]

        widths_constant = np.copy(self.age_widths)

        if self.age_lhs[age_ind] - par_dict["age"]*10**9 > 0:
            widths_constant[age_ind] = 0.
            widths_constant[age_ind-1] = par_dict["age"]*10**9 - self.age_lhs[age_ind-1]
            
        else:
            widths_constant[age_ind] = par_dict["age"]*10**9 - self.age_lhs[age_ind]

        if self.age_lhs[age_min_ind] - par_dict["age_min"]*10**9 < 0:
            widths_constant[age_min_ind-1] = 0.
            widths_constant[age_min_ind] = self.age_lhs[age_min_ind+1] - par_dict["age_min"]*10**9
            
        else:
            widths_constant[age_min_ind-1] = self.age_lhs[age_min_ind] - par_dict["age_min"]*10**9

        if age_min_ind > 0:
            widths_constant[:age_min_ind-1] *= 0.

        widths_constant[age_ind+1:] *= 0.

        if age_ind == age_min_ind and self.age_lhs[age_ind] - par_dict["age"]*10**9 < 0 and self.age_lhs[age_min_ind] - par_dict["age_min"]*10**9 < 0:
            widths_constant[age_ind] = (par_dict["age"] - par_dict["age_min"])*10**9

        weight_widths = widths_constant
        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["mass"]
        
        return weight_widths



    """ returns an array of sfr(t) values for a delayed component normalised to one Solar mass of star formation by the present time. """
    def delayed(self, par_dict):

        age_ind = self.ages[self.ages < par_dict["age"]*10**9].shape[0]

        widths_delayed = np.copy(self.age_widths)
       
        weights_delayed = np.zeros(len(widths_delayed))
        weights_delayed[:age_ind+1] = (self.ages[age_ind+1] - self.ages[:age_ind+1])*np.exp(-(self.ages[age_ind+1] - self.ages[:age_ind+1])/(10**par_dict["tau"]*10**9))
        
        if self.age_lhs[age_ind] - par_dict["age"]*10**9 > 0:
            widths_delayed[age_ind] = 0.
            widths_delayed[age_ind-1] = par_dict["age"]*10**9 - self.age_lhs[age_ind-1]
            
        else:
            widths_delayed[age_ind] = par_dict["age"]*10**9 - self.age_lhs[age_ind]

        weight_widths = weights_delayed*widths_delayed
        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["mass"]
        
        return weight_widths



    """ returns an array of sfr(t) values for a lognormal component normalised to one Solar mass of star formation by the present time. """
    def lognormal(self, par_dict):
        
        age_at_zred = np.interp(self.model_components["zred"], z_array, age_at_z)*10**9

        if "tmax" in par_dict.keys() and "fwhm" in par_dict.keys():
            tmax, fwhm = par_dict["tmax"]*10**9, par_dict["fwhm"]*10**9

            if tmax < 10**8:
                sys.exit("BAGPIPES: Time of max star formation for lognormal component cannot be before the Universe is 100Myr old.")

            tau, T0 = fsolve(lognorm_equations, (fwhm/(2*tmax*np.sqrt(2*np.log(2))), np.log(tmax) + fwhm**2/(8*np.log(2)*tmax**2)), args=([tmax, fwhm]))

        else:
            tau, T0 = par_dict["tau"], par_dict["T0"]

        age_lhs_universe = age_at_zred - self.age_lhs
        ages_universe = age_at_zred - self.ages[age_lhs_universe > 0.]

        width_red_factor = (age_at_zred - self.age_lhs[age_lhs_universe > 0.][-1])/(self.age_lhs[age_lhs_universe[age_lhs_universe > 0.].shape[0]] - self.age_lhs[age_lhs_universe > 0.][-1])
        
        ages_universe[-1] = age_lhs_universe[age_lhs_universe > 0.][-1]/2.

        log_ages_universe = np.log(ages_universe)

        widths_lognormal = np.copy(self.age_widths)

        widths_lognormal[-1] *= width_red_factor
       
        weights_lognormal = np.zeros(widths_lognormal.shape[0])

        weights_lognormal[:log_ages_universe.shape[0]] = (1./np.sqrt(2*np.pi*tau**2))*(1./ages_universe)*np.exp(-(log_ages_universe - T0)**2/(2*tau**2))

        weight_widths = weights_lognormal*widths_lognormal

        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["mass"]

        return weight_widths



    """ returns an array of sfr(t) values for a lognormal component normalised to one Solar mass of star formation by the present time. """
    def lognormalgauss(self, par_dict):
        
        age_at_zred = np.interp(self.model_components["zred"], z_array, age_at_z)*10**9

        if "tmax" in par_dict.keys() and "fwhm" in par_dict.keys():
            tmax, fwhm = par_dict["tmax"]*10**9, par_dict["fwhm"]*10**9

            if tmax < 10**8:
                sys.exit("BAGPIPES: Time of max star formation for lognormal component cannot be before the Universe is 100Myr old.")

            tau, T0 = fsolve(lognorm_equations, (fwhm/(2*tmax*np.sqrt(2*np.log(2))), np.log(tmax) + fwhm**2/(8*np.log(2)*tmax**2)), args=([tmax, fwhm]))

        else:
            tau, T0 = par_dict["tau"], par_dict["T0"]

        age_lhs_universe = age_at_zred - self.age_lhs
        ages_universe = age_at_zred - self.ages[age_lhs_universe > 0.]

        width_red_factor = (age_at_zred - self.age_lhs[age_lhs_universe > 0.][-1])/(self.age_lhs[age_lhs_universe[age_lhs_universe > 0.].shape[0]] - self.age_lhs[age_lhs_universe > 0.][-1])
        
        ages_universe[-1] = age_lhs_universe[age_lhs_universe > 0.][-1]/2.

        log_ages_universe = np.log(ages_universe)

        widths_lognormal = np.copy(self.age_widths)

        widths_lognormal[-1] *= width_red_factor
       
        weights_lognormal = np.zeros(widths_lognormal.shape[0])

        weights_lognormal[:log_ages_universe.shape[0]] = (1./np.sqrt(2*np.pi*tau**2))*(1./ages_universe)*np.exp(-(log_ages_universe - T0)**2/(2*tau**2))


        weights_gausscomp = np.zeros(widths_lognormal.shape[0])
        weights_gausscomp[:log_ages_universe.shape[0]] = np.exp(-self.ages[age_lhs_universe > 0.]**2/(par_dict["gausssig"]*10**9)**2)
        weight_widths_gausscomp = weights_gausscomp*widths_lognormal
        weight_widths_gausscomp /= np.sum(weight_widths_gausscomp)
        weight_widths_gausscomp *= 10**par_dict["gaussmass"]


        weight_widths = weights_lognormal*widths_lognormal

        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["mass"]

        return weight_widths + weight_widths_gausscomp



    """ returns an array of sfr(t) values for a double power law component normalised to one Solar mass of star formation by the present time. """
    def dblplaw(self, par_dict):

        alpha = par_dict["alpha"]
        beta = par_dict["beta"]
        tau = par_dict["tau"]*10**9
        
        age_at_zred = np.interp(self.model_components["zred"], z_array, age_at_z)*10**9

        age_lhs_universe = age_at_zred - self.age_lhs
        ages_universe = age_at_zred - self.ages[age_lhs_universe > 0.]

        width_red_factor = (age_at_zred - self.age_lhs[age_lhs_universe > 0.][-1])/(self.age_lhs[age_lhs_universe[age_lhs_universe > 0.].shape[0]] - self.age_lhs[age_lhs_universe > 0.][-1])

        ages_universe[-1] = age_lhs_universe[age_lhs_universe > 0.][-1]/2.

        widths_dblplaw = np.copy(self.age_widths)

        widths_dblplaw[-1] *= width_red_factor
       
        weights_dblplaw = np.zeros(widths_dblplaw.shape[0])

        weights_dblplaw[:ages_universe.shape[0]] = ((ages_universe/tau)**alpha + (ages_universe/tau)**-beta)**-1

        weight_widths = weights_dblplaw*widths_dblplaw

        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["mass"]

        return weight_widths


    def custom(self, par_dict):

        age_at_zred = np.interp(self.model_components["zred"], z_array, age_at_z)*10**9

        if isinstance(par_dict["history"], str):
            ages = np.loadtxt(par_dict["history"], usecols=(0,))
            sfr = np.loadtxt(par_dict["history"], usecols=(1,))

        else:
            ages = par_dict["history"][:,0]
            sfr = par_dict["history"][:,1]

        weights_custom = np.interp(self.ages, ages, sfr, right=0.)

        weight_widths = weights_custom*np.copy(self.age_widths)

        return weight_widths


    """ Creates a plot of sfr(t) normalised to one Solar mass of star formation by the present time. """
    def plot(self):

        plt.figure()
        plt.plot(self.ages*10**-9, self.sfr) 
        plt.ylabel("SFR ($\mathrm{M_\odot\ yr^{-1}}$)")
        plt.xlabel("Time before observation (Gyr)")
        plt.ylim(0, 1.1*np.max(self.sfr))
        plt.xlim(0, 1.05*self.maxage)
        plt.show()
        

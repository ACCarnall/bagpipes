import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

import model_manager as models
import model_galaxy

component_types = ["burst", "constant", "exponential", "cexp", "delayed", "lognormal", "dblplaw", "lognormalgauss", "custom"]
special_component_types = ["lognormal", "dblplaw", "lognormalgauss", "custom", "cexp"]



def lognorm_equations(p, consts):

    tau_solve, T0_solve = p

    xmax, h = consts

    return (np.exp(T0_solve - tau_solve**2) - xmax, xmax*(np.exp(0.5*np.sqrt(8*np.log(2)*tau_solve**2)) - np.exp(-0.5*np.sqrt(8*np.log(2)*tau_solve**2))) - h)



class Star_Formation_History:
    """ Generate a star formation history. This class is not intended to be used directly, it should be accessed through Model_Galaxy.

    Parameters
    ----------

    model_components : dict
        A dictionary containing information about the star formation history you wish to generate. 

    """

    def __init__(self, model_components):

        models.set_cosmology()

        self.ages = models.chosen_ages
        self.age_lhs = models.chosen_age_lhs[:-1]
        self.age_widths = models.chosen_age_widths

        # component_types: List of all possible types of star formation history (SFH) components
        self.component_types = component_types

        self.special_component_types = special_component_types

        # model_components: dictionary containing all information about the model being generated    
        self.model_components = model_components

        # weight_widths: A dictionary which will contain arrays of SSP weights for different SFH components to allow CSPs to be made
        self.weight_widths = {}
        
        # populate the weight_widths dict with arrays of ssfr values by calling the functions corresponding to the SFH component type
        for component in list(self.model_components):
            if component in self.component_types or component[:-1] in self.component_types:
                for option in self.component_types:
                    if component[:len(option)] == option:
                        self.weight_widths[component] = getattr(self, option)(self.model_components[component])

        # sfr: an array of total (mass normalised) star formation rate values across all SFH components 
        self.sfr = np.zeros(len(self.ages))

        # sum all of the weight_widths to get the total sfr(t)
        for component in list(self.model_components):
            if component in self.component_types or component[:-1] in self.component_types:
                self.sfr += self.weight_widths[component]/self.age_widths

        # obtain the maximum age (time before observation) for which the ssfr is non-zero in Gyr (for plotting purposes)
        self.maxage = 0.

        if np.max(np.isin(list(self.model_components), self.special_component_types)):
            self.maxage = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)

        # Hacky prior on dblplaw:tau
        if "dblplaw" in list(self.model_components):
            if self.model_components["dblplaw"]["tau"] > self.maxage:
                self.maxage = self.model_components["dblplaw"]["tau"]
        
        for component in list(self.model_components):
            if component in self.component_types or component[:-1] in self.component_types:
                if component not in self.special_component_types and component[:-1] not in self.special_component_types:
                    if model_components[component]["age"] == "hubble time":
                        self.maxage = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)
                                        
                    if model_components[component]["age"] > self.maxage:
                        self.maxage = model_components[component]["age"]




    """ returns an array of sfr(t) values for a burst component. """
    def burst(self, par_dict):

        # If a burst with finite width is requested, set up a constant component between the start and finish times.
        if "width" in list(par_dict):
            if par_dict["age"] - par_dict["width"] < 0:
                sys.exit("BAGPIPES: The time at which the burst started was less than zero.")

            const_par_dict = {}
            const_par_dict["agemax"] = par_dict["age"]
            const_par_dict["agemin"] = par_dict["age"] - par_dict["width"]
            const_par_dict["massformed"] = par_dict["massformed"]

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
            weight_widths *= 10**par_dict["massformed"]

            return weight_widths



    """ returns an array of sfr(t) values for an exponential component. """
    def exponential(self, par_dict):
        
        age = par_dict["age"]*10**9
        if "tau" in list(par_dict) and "nefolds" in list(par_dict):
            sys.exit("BAGPIPES: For exponential component please specify one of tau or nefolds only.")

        if "nefolds" in list(par_dict):    
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
        weight_widths *= 10**par_dict["massformed"]

        return weight_widths



    """ returns an array of sfr(t) values for a cexp component: constant from beginning of Universe until T0, after which exp decline. """
    def cexp(self, par_dict):
        
        T0 = par_dict["T0"]*10**9
        tau = par_dict["tau"]*10**9
        
        models.age_at_zred = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)*10**9

        age_lhs_universe = models.age_at_zred - self.age_lhs
        ages_universe = models.age_at_zred - self.ages[age_lhs_universe > 0.]

        width_red_factor = (models.age_at_zred - self.age_lhs[age_lhs_universe > 0.][-1])/(self.age_lhs[age_lhs_universe[age_lhs_universe > 0.].shape[0]] - self.age_lhs[age_lhs_universe > 0.][-1])

        ages_universe[-1] = age_lhs_universe[age_lhs_universe > 0.][-1]/2.

        widths_cexp = np.copy(self.age_widths)

        widths_cexp[-1] *= width_red_factor
       
        weights_cexp = np.zeros(widths_cexp.shape[0])

        weights_cexp[:ages_universe.shape[0]] = 1.
        weights_cexp[:ages_universe.shape[0]][ages_universe > T0] *= np.exp(-(ages_universe[ages_universe > T0] - T0)/tau)

        weight_widths = weights_cexp*widths_cexp

        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["massformed"]

        return weight_widths



    """ returns an array of sfr(t) values for a constant component. """
    def constant(self, par_dict):
        
        if par_dict["age"] == "hubble time":
            par_dict["age"] = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)

        if par_dict["agemin"] > par_dict["agemax"]:
            sys.exit("Minimum constant age exceeded maximum.")
        
        age_ind = self.ages[self.ages < par_dict["age"]*10**9].shape[0]

        age_min_ind = self.ages[self.ages < par_dict["agemin"]*10**9].shape[0]

        widths_constant = np.copy(self.age_widths)

        if self.age_lhs[age_ind] - par_dict["agemax"]*10**9 > 0:
            widths_constant[age_ind] = 0.
            widths_constant[age_ind-1] = par_dict["agemax"]*10**9 - self.age_lhs[age_ind-1]
            
        else:
            widths_constant[age_ind] = par_dict["agemax"]*10**9 - self.age_lhs[age_ind]

        if self.age_lhs[age_min_ind] - par_dict["agemin"]*10**9 < 0:
            widths_constant[age_min_ind-1] = 0.
            widths_constant[age_min_ind] = self.age_lhs[age_min_ind+1] - par_dict["agemin"]*10**9
            
        else:
            widths_constant[age_min_ind-1] = self.age_lhs[age_min_ind] - par_dict["agemin"]*10**9

        if age_min_ind > 0:
            widths_constant[:age_min_ind-1] *= 0.

        widths_constant[age_ind+1:] *= 0.

        if age_ind == age_min_ind and self.age_lhs[age_ind] - par_dict["agemax"]*10**9 < 0 and self.age_lhs[age_min_ind] - par_dict["agemin"]*10**9 < 0:
            widths_constant[age_ind] = (par_dict["agemax"] - par_dict["agemin"])*10**9

        weight_widths = widths_constant
        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["massformed"]
        
        return weight_widths



    """ returns an array of sfr(t) values for a delayed component. """
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
        weight_widths *= 10**par_dict["massformed"]
        
        return weight_widths



    """ returns an array of sfr(t) values for a lognormal component. """
    def lognormal(self, par_dict):
        
        models.age_at_zred = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)*10**9

        if "tmax" in list(par_dict) and "fwhm" in list(par_dict):
            tmax, fwhm = par_dict["tmax"]*10**9, par_dict["fwhm"]*10**9

            if tmax < 10**8:
                sys.exit("BAGPIPES: Time of max star formation for lognormal component cannot be before the Universe is 100Myr old.")

            tau, T0 = fsolve(lognorm_equations, (fwhm/(2*tmax*np.sqrt(2*np.log(2))), np.log(tmax) + fwhm**2/(8*np.log(2)*tmax**2)), args=([tmax, fwhm]))

        else:
            tau, T0 = par_dict["tau"], par_dict["T0"]

        age_lhs_universe = models.age_at_zred - self.age_lhs
        ages_universe = models.age_at_zred - self.ages[age_lhs_universe > 0.]

        width_red_factor = (models.age_at_zred - self.age_lhs[age_lhs_universe > 0.][-1])/(self.age_lhs[age_lhs_universe[age_lhs_universe > 0.].shape[0]] - self.age_lhs[age_lhs_universe > 0.][-1])
        
        ages_universe[-1] = age_lhs_universe[age_lhs_universe > 0.][-1]/2.

        log_ages_universe = np.log(ages_universe)

        widths_lognormal = np.copy(self.age_widths)

        widths_lognormal[-1] *= width_red_factor
       
        weights_lognormal = np.zeros(widths_lognormal.shape[0])

        weights_lognormal[:log_ages_universe.shape[0]] = (1./np.sqrt(2*np.pi*tau**2))*(1./ages_universe)*np.exp(-(log_ages_universe - T0)**2/(2*tau**2))

        weight_widths = weights_lognormal*widths_lognormal

        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["massformed"]

        return weight_widths



    """ returns an array of sfr(t) values for a lognormal component with a gaussian stuck on the end. """
    def lognormalgauss(self, par_dict):
        
        models.age_at_zred = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)*10**9

        if "tmax" in list(par_dict) and "fwhm" in list(par_dict):
            tmax, fwhm = par_dict["tmax"]*10**9, par_dict["fwhm"]*10**9

            if tmax < 10**8:
                sys.exit("BAGPIPES: Time of max star formation for lognormal component cannot be before the Universe is 100Myr old.")

            tau, T0 = fsolve(lognorm_equations, (fwhm/(2*tmax*np.sqrt(2*np.log(2))), np.log(tmax) + fwhm**2/(8*np.log(2)*tmax**2)), args=([tmax, fwhm]))

        else:
            tau, T0 = par_dict["tau"], par_dict["T0"]

        age_lhs_universe = models.age_at_zred - self.age_lhs
        ages_universe = models.age_at_zred - self.ages[age_lhs_universe > 0.]

        width_red_factor = (models.age_at_zred - self.age_lhs[age_lhs_universe > 0.][-1])/(self.age_lhs[age_lhs_universe[age_lhs_universe > 0.].shape[0]] - self.age_lhs[age_lhs_universe > 0.][-1])
        
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
        weight_widths *= 10**par_dict["massformed"]

        return weight_widths + weight_widths_gausscomp



    """ returns an array of sfr(t) values for a double power law component. """
    def dblplaw(self, par_dict):

        alpha = par_dict["alpha"]
        beta = par_dict["beta"]
        tau = par_dict["tau"]*10**9
        
        models.age_at_zred = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)*10**9

        age_lhs_universe = models.age_at_zred - self.age_lhs
        ages_universe = models.age_at_zred - self.ages[age_lhs_universe > 0.]

        width_red_factor = (models.age_at_zred - self.age_lhs[age_lhs_universe > 0.][-1])/(self.age_lhs[age_lhs_universe[age_lhs_universe > 0.].shape[0]] - self.age_lhs[age_lhs_universe > 0.][-1])

        ages_universe[-1] = age_lhs_universe[age_lhs_universe > 0.][-1]/2.

        widths_dblplaw = np.copy(self.age_widths)

        widths_dblplaw[-1] *= width_red_factor
       
        weights_dblplaw = np.zeros(widths_dblplaw.shape[0])

        weights_dblplaw[:ages_universe.shape[0]] = ((ages_universe/tau)**alpha + (ages_universe/tau)**-beta)**-1

        weight_widths = weights_dblplaw*widths_dblplaw

        weight_widths /= np.sum(weight_widths)
        weight_widths *= 10**par_dict["massformed"]

        return weight_widths



    def custom(self, par_dict):

        models.age_at_zred = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)*10**9

        if isinstance(par_dict["history"], str):
            ages = np.loadtxt(par_dict["history"], usecols=(0,))
            sfr = np.loadtxt(par_dict["history"], usecols=(1,))

        else:
            ages = par_dict["history"][:,0]*10**9
            sfr = par_dict["history"][:,1]

        weights_custom = np.interp(self.ages, ages, sfr, right=0.)

        weight_widths = weights_custom*np.copy(self.age_widths)

        return weight_widths


    """ Creates a plot of sfr(t) for this star formation history. """
    def plot(self):

        models.age_at_zred = np.interp(self.model_components["redshift"], models.z_array, models.age_at_z)*10**9

        sfh_x = np.zeros(2*self.ages.shape[0])
        sfh_y = np.zeros(2*self.sfr.shape[0])

        for j in range(self.sfr.shape[0]):

            sfh_x[2*j] = self.age_lhs[j]
            if j+1 < self.sfr.shape[0]:
                sfh_x[2*j + 1] = self.age_lhs[j+1]

            sfh_y[2*j] = self.sfr[j]
            sfh_y[2*j + 1] = self.sfr[j]

        sfh_x[-2:] = 1.5*10**10

        plt.figure(figsize=(12, 4))
        plt.plot((models.age_at_zred - sfh_x)*10**-9, sfh_y, color="black", lw=1.5)
        plt.ylabel("$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")
        plt.xlabel("$\mathrm{Age\ of\ Universe\ /\ Gyr}$")

        #plt.xlabel("$\mathrm{Time\ before\ observation\ (Gyr)}$")
        plt.ylim(0, 1.1*np.max(self.sfr))
        plt.xlim(models.age_at_zred*10**-9, 0.)
        #plt.savefig("examplesfh.jpg", bbox_inches="tight")
        plt.show()
        

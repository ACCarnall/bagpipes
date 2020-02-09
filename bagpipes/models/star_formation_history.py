from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.optimize import fsolve
from copy import copy, deepcopy

from .. import utils
from .. import config
from .. import plotting

from .chemical_enrichment_history import chemical_enrichment_history


def lognorm_equations(p, consts):
    """ Equations for finding the tau and T0 for a lognormal SFH given
    some tmax and FWHM. Needed to transform variables. """

    tau_solve, T0_solve = p

    xmax, h = consts

    tau = np.exp(T0_solve - tau_solve**2) - xmax
    t0 = xmax*(np.exp(0.5*np.sqrt(8*np.log(2)*tau_solve**2))
               - np.exp(-0.5*np.sqrt(8*np.log(2)*tau_solve**2))) - h

    return (tau, t0)


class star_formation_history:
    """ Generate a star formation history.

    Parameters
    ----------

    model_components : dict
        A dictionary containing information about the star formation
        history you wish to generate.

    log_sampling : float - optional
        the log of the age sampling of the SFH, defaults to 0.0025.
    """

    def __init__(self, model_components, log_sampling=0.0025):

        self.hubble_time = utils.age_at_z[utils.z_array == 0.]

        # Set up the age sampling for internal SFH calculations.
        log_age_max = np.log10(self.hubble_time)+9. + 2*log_sampling
        self.ages = np.arange(6., log_age_max, log_sampling)
        self.age_lhs = utils.make_bins(self.ages, make_rhs=True)[0]
        self.ages = 10**self.ages
        self.age_lhs = 10**self.age_lhs
        self.age_lhs[0] = 0.
        self.age_lhs[-1] = 10**9*self.hubble_time
        self.age_widths = self.age_lhs[1:] - self.age_lhs[:-1]

        # Detect SFH components
        comp_list = list(model_components)
        self.components = ([k for k in comp_list if k in dir(self)]
                           + [k for k in comp_list if k[:-1] in dir(self)])

        self.component_sfrs = {}  # SFR versus time for all components.
        self.component_weights = {}  # SSP weights for all components.

        self._resample_live_frac_grid()

        self.update(model_components)

    def update(self, model_components):

        self.model_components = model_components
        self.redshift = self.model_components["redshift"]

        self.sfh = np.zeros_like(self.ages)  # Star-formation history

        self.unphysical = False
        self.age_of_universe = 10**9*np.interp(self.redshift, utils.z_array,
                                               utils.age_at_z)

        # Calculate the star-formation history for each of the components.
        for i in range(len(self.components)):

            name = self.components[i]
            func = self.components[i]

            if name not in dir(self):
                func = name[:-1]

            self.component_sfrs[name] = np.zeros_like(self.ages)
            self.component_weights[name] = np.zeros_like(config.age_sampling)

            getattr(self, func)(self.component_sfrs[name],
                                self.model_components[name])

            # Normalise to the correct mass.
            mass_norm = np.sum(self.component_sfrs[name]*self.age_widths)
            desired_mass = 10**self.model_components[name]["massformed"]

            self.component_sfrs[name] *= desired_mass/mass_norm
            self.sfh += self.component_sfrs[name]

            # Sum up contributions to each age bin to create SSP weights
            weights = self.component_sfrs[name]*self.age_widths
            self.component_weights[name] = np.histogram(self.ages,
                                                        bins=config.age_bins,
                                                        weights=weights)[0]
        # Check no stars formed before the Big Bang.
        if self.sfh[self.ages > self.age_of_universe].max() > 0.:
            self.unphysical = True

        # ceh: Chemical enrichment history object
        self.ceh = chemical_enrichment_history(self.model_components,
                                               self.component_weights)

        self._calculate_derived_quantities()

    def _calculate_derived_quantities(self):
        self.stellar_mass = np.log10(np.sum(self.live_frac_grid*self.ceh.grid))
        self.formed_mass = np.log10(np.sum(self.ceh.grid))

        age_mask = (self.ages < 10**8)
        self.sfr = np.sum(self.sfh[age_mask]*self.age_widths[age_mask])
        self.sfr /= self.age_widths[age_mask].sum()
        self.ssfr = np.log10(self.sfr) - self.stellar_mass
        self.nsfr = np.log10(self.sfr*self.age_of_universe) - self.formed_mass

        self.mass_weighted_age = np.sum(self.sfh*self.age_widths*self.ages)
        self.mass_weighted_age /= np.sum(self.sfh*self.age_widths)

        self.tform = self.age_of_universe - self.mass_weighted_age

        self.tform *= 10**-9
        self.mass_weighted_age *= 10**-9

        mass_assembly = np.cumsum(self.sfh[::-1]*self.age_widths[::-1])[::-1]
        tunivs = self.age_of_universe - self.ages
        mean_sfrs = mass_assembly/tunivs
        normed_sfrs = np.zeros_like(self.sfh)
        sf_mask = (self.sfh > 0.)
        normed_sfrs[sf_mask] = self.sfh[sf_mask]/mean_sfrs[sf_mask]

        if self.sfr > 0.1*mean_sfrs[0]:
            self.tquench = 99.

        else:
            quench_ind = np.argmax(normed_sfrs > 0.1)
            self.tquench = tunivs[quench_ind]*10**-9

    def _resample_live_frac_grid(self):
        self.live_frac_grid = np.zeros((config.metallicities.shape[0],
                                        config.age_sampling.shape[0]))

        raw_live_frac_grid = config.live_frac

        for i in range(config.metallicities.shape[0]):
            self.live_frac_grid[i, :] = np.interp(config.age_sampling,
                                                  config.raw_stellar_ages,
                                                  raw_live_frac_grid[:, i])

    def burst(self, sfr, param):
        """ A delta function burst of star-formation. """

        if "age" in list(param):
            age = param["age"]*10**9

        elif "tform" in list(param):
            age = self.age_of_universe - param["tform"]*10**9

        sfr[np.argmin(np.abs(self.ages - age))] += 1

    def constant(self, sfr, param):
        """ Constant star-formation between some limits. """

        if "age_min" in list(param):
            if param["age_max"] == "age_of_universe":
                age_max = self.age_of_universe

            else:
                age_max = param["age_max"]*10**9

            age_min = param["age_min"]*10**9

        else:
            age_max = self.age_of_universe - param["tstart"]*10**9
            age_min = self.age_of_universe - param["tstop"]*10**9

        mask = (self.ages > age_min) & (self.ages < age_max)
        sfr[mask] += 1.

    def exponential(self, sfr, param):

        if "age" in list(param):
            age = param["age"]*10**9

        else:
            age = (param["tstart"] - self.age_of_universe)*10**9

        tau = param["tau"]*10**9

        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = np.exp(-t/tau)

    def delayed(self, sfr, param):

        age = param["age"]*10**9
        tau = param["tau"]*10**9

        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = t*np.exp(-t/tau)

    def const_exp(self, sfr, param):

        age = param["age"]*10**9
        tau = param["tau"]*10**9

        t = age - self.ages[self.ages < age]

        sfr[self.ages < age] = np.exp(-t/tau)
        sfr[(self.ages > age) & (self.ages < self.age_of_universe)] = 1.

    def lognormal(self, sfr, param):
        if "tmax" in list(param) and "fwhm" in list(param):
            tmax, fwhm = param["tmax"]*10**9, param["fwhm"]*10**9

            tau_guess = fwhm/(2*tmax*np.sqrt(2*np.log(2)))
            t0_guess = np.log(tmax) + fwhm**2/(8*np.log(2)*tmax**2)

            tau, t0 = fsolve(lognorm_equations, (tau_guess, t0_guess),
                             args=([tmax, fwhm]))

        else:
            tau, t0 = par_dict["tau"], par_dict["t0"]

        mask = self.ages < self.age_of_universe
        t = self.age_of_universe - self.ages[mask]

        sfr[mask] = ((1./np.sqrt(2.*np.pi*tau**2))*(1./t)
                     * np.exp(-(np.log(t) - t0)**2/(2*tau**2)))

    def dblplaw(self, sfr, param):
        alpha = param["alpha"]
        beta = param["beta"]
        tau = param["tau"]*10**9

        mask = self.ages < self.age_of_universe
        t = self.age_of_universe - self.ages[mask]

        sfr[mask] = ((t/tau)**alpha + (t/tau)**-beta)**-1

        if tau > self.age_of_universe:
            self.unphysical = True

    def custom(self, sfr, param):
        history = param["history"]
        if isinstance(history, str):
            custom_sfh = np.loadtxt(history)

        else:
            custom_sfh = history

        sfr[:] = np.interp(self.ages, custom_sfh[:, 0], custom_sfh[:, 1],
                           left=0, right=0)

        sfr[self.ages > self.age_of_universe] = 0.

    def plot(self, show=True):
        return plotting.plot_sfh(self, show=show)

from __future__ import print_function, division, absolute_import

import numpy as np

from copy import deepcopy

from .fit import fit_info_parser
from .star_formation_history import star_formation_history

from . import plotting
from . import utils


class check_prior(fit_info_parser):

    def __init__(self, fit_instructions, n_draws=10000, name="", run="."):

        fit_info_parser.__init__(self, fit_instructions)

        self.n_draws = n_draws

        # name: string to appear in plots identifying this prior
        self.name = name

        self.draw_sfh()
        self._set_up_prior_dict()

        for i in range(self.n_draws):

            self.draw_sfh()

            for name in self.fit_params:
                split = name.split(":")

                if len(split) == 1:
                    self.prior[name][i] = self.model_comp[split[0]]

                elif len(split) == 2:
                    self.prior[name][i] = self.model_comp[split[0]][split[1]]

            self.prior["sfr"][i] = self.sfh.sfr_100myr
            self.prior["mwa"][i] = 10**-9*self.sfh.mass_weighted_age
            self.prior["tmw"][i] = 10**-9*(self.sfh.age_of_universe
                                           - self.sfh.mass_weighted_age)

            if "redshift" in self.fixed_params:
                self.prior["sfh"][i, :] = self.sfh.sfr["total"]

            mtot = self.sfh.mass["total"]
            self.prior["mass"]["total"]["formed"][i] = mtot["formed"]
            self.prior["mass"]["total"]["living"][i] = mtot["living"]

        self.prior["ssfr"] = np.log10(self.prior["sfr"]/mtot["living"])

    def _set_up_prior_dict(self):

        self.prior = {}

        for name in self.fit_params:
            self.prior[name] = np.zeros(self.n_draws)

        self.prior["mwa"] = np.zeros(self.n_draws)
        self.prior["sfr"] = np.zeros(self.n_draws)
        self.prior["tmw"] = np.zeros(self.n_draws)

        if "redshift" in self.fixed_params:
            self.prior["sfh"] = np.zeros((self.n_draws,
                                          self.sfh.ages.shape[0]))

        self.prior["mass"] = {}
        self.prior["mass"]["total"] = {}
        self.prior["mass"]["total"]["living"] = np.zeros(self.n_draws)
        self.prior["mass"]["total"]["formed"] = np.zeros(self.n_draws)

    def draw_sfh(self):
        unphysical = True

        while unphysical:
            cube = np.random.rand(self.ndim)
            param = self._prior_transform(cube)

            self.model_comp = self._get_model_comp(param)
            self.sfh = star_formation_history(self.model_comp)
            unphysical = self.sfh.unphysical

    def plot_1d(self, show=True, save=False):
        plotting.plot_1d_distributions(self, show=show, save=save)

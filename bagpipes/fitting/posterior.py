from __future__ import print_function, division, absolute_import

import numpy as np

import os
import deepdish as dd

from .fitted_model import fitted_model

from ..models.star_formation_history import star_formation_history
from ..models.model_galaxy import model_galaxy


class posterior(object):
    """ Provides access to the outputs from fitting models to data and
    calculating posterior predictions for derived parameters (e.g. for
    star-formation histories, rest-frane magnitudes etc).

    Parameters
    ----------
    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    n_samples : float - optional
        The number of posterior samples to generate for each quantity.
    """

    def __init__(self, galaxy, run=".", n_samples=500):

        self.galaxy = galaxy
        self.run = run
        self.n_samples = n_samples

        fname = "pipes/posterior/" + self.run + "/" + self.galaxy.ID + ".h5"

        # Check to see whether the object has been fitted.
        if not os.path.exists(fname):
            raise IOError("Fit results not found for " + self.galaxy.ID + ".")

        # Reconstruct the fitted model.
        self.fit_instructions = dd.io.load(fname, group="/fit_instructions")
        self.fitted_model = fitted_model(self.galaxy, self.fit_instructions)

        # 2D array of samples for the fitted parameters only.
        self.samples2d = dd.io.load(fname, group="/samples2d")

        self.samples = {}  # Store all posterior samples

        for i in range(self.fitted_model.ndim):
            self.samples[self.fitted_model.params[i]] = self.samples2d[:,i]

        self.get_basic_quantities()
        self.get_advanced_quantities()

    def get_basic_quantities(self):
        """Calculates basic derived posterior quantities, these are fast
        as they are derived only from the star-formation history """

        self.fitted_model._update_model_components(self.samples2d[0, :])
        self.sfh = star_formation_history(self.fitted_model.model_components)

        quantity_names = ["stellar_mass", "formed_mass", "sfr", "ssfr",
                          "mass_weighted_age", "tform", "tquench"]

        for q in quantity_names:
            self.samples[q] = np.zeros(self.n_samples)

        self.samples["sfh"] = np.zeros((self.n_samples,
                                        self.sfh.ages.shape[0]))

        quantity_names += ["sfh"]

        for i in range(self.n_samples):
            self.fitted_model._update_model_components(self.samples2d[i, :])
            self.sfh.update(self.fitted_model.model_components)

            for q in quantity_names:
                self.samples[q][i] = getattr(self.sfh, q)

    def get_advanced_quantities(self):
        """Calculates advanced derived posterior quantities, these are
        slower because they require the full model spectra. """

        self.fitted_model._update_model_components(self.samples2d[0, :])
        self.model_galaxy = model_galaxy(self.fitted_model.model_components,
                                         filt_list=self.galaxy.filt_list,
                                         spec_wavs=self.galaxy.spec_wavs)

        all_names = ["photometry", "uvj", "spectrum", "spectrum_full"]
        all_model_keys = dir(self.model_galaxy)
        quantity_names = [q for q in all_names if q in all_model_keys]

        for q in quantity_names:
            size = getattr(self.model_galaxy, q).shape[0]
            self.samples[q] = np.zeros((self.n_samples, size))

        if "polynomial" in list(self.fitted_model.model_components):
            size = self.model_galaxy.spectrum.shape[0]
            self.samples["polynomial"] = np.zeros((self.n_samples, size))

        for i in range(self.n_samples):
            self.fitted_model._update_model_components(self.samples2d[i, :])
            self.model_galaxy.update(self.fitted_model.model_components)

            if "polynomial" in list(self.fitted_model.model_components):
                self.fitted_model._update_polynomial()
                self.samples["polynomial"][i] = self.fitted_model.polynomial

            for q in quantity_names:
                self.samples[q][i] = getattr(self.model_galaxy, q)

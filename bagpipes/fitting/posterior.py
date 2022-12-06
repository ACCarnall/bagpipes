from __future__ import print_function, division, absolute_import

import numpy as np

import os
import deepdish as dd

from copy import deepcopy

from .fitted_model import fitted_model
from .prior import dirichlet

from ..models.star_formation_history import star_formation_history
from ..models.model_galaxy import model_galaxy

from .. import utils


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

        # If fewer than n_samples exist in posterior, reduce n_samples
        if self.samples2d.shape[0] < self.n_samples:
            self.n_samples = self.samples2d.shape[0]

        # Randomly choose points to generate posterior quantities
        self.indices = np.random.choice(self.samples2d.shape[0],
                                        size=self.n_samples, replace=False)

        self.samples = {}  # Store all posterior samples

        dirichlet_comps = []  # Do any parameters follow Dirichlet dist

        # Add 1D posteriors for fitted params to the samples dictionary
        for i in range(self.fitted_model.ndim):
            param_name = self.fitted_model.params[i]

            if "dirichlet" in param_name:
                dirichlet_comps.append(param_name.split(":")[0])

            self.samples[param_name] = self.samples2d[self.indices, i]

        self.get_dirichlet_tx(dirichlet_comps)

        self.get_basic_quantities()

    def get_dirichlet_tx(self, dirichlet_comps):
        """ Calculate tx vals for any Dirichlet distributed params. """

        if len(dirichlet_comps) > 0:
            comp = dirichlet_comps[0]

            r_samples = []

            # Pull out the values of the fitted "r" parameters
            for i in range(len(self.fitted_model.params)):
                split = self.fitted_model.params[i].split(":")
                if (split[0] == comp) and ("dirichlet" in split[1]):
                    r_samples.append(self.samples2d[:, i])

            r_values = np.c_[r_samples].T
            self.samples[comp + ":r"] = r_values
            self.samples[comp + ":tx"] = np.zeros_like(r_values)

            # Convert the fitted "r" params into tx values.
            for i in range(self.samples2d.shape[0]):
                alpha = self.fit_instructions[comp]["alpha"]
                r = self.samples[comp + ":r"][i, :]
                self.samples[comp + ":tx"][i, :] = dirichlet(r, alpha)[:-1]

            # Get the age of the Universe to convert tx into Gyr.
            if "redshift" in self.fitted_model.params:
                redshift_ind = self.fitted_model.params.index("redshift")
                age_of_universe = np.interp(self.samples2d[:, redshift_ind],
                                            utils.z_array, utils.age_at_z)
                age_of_universe = np.expand_dims(age_of_universe, axis=1)

            else:
                age_of_universe = np.interp(self.fit_instructions["redshift"],
                                            utils.z_array, utils.age_at_z)

            self.samples[comp + ":tx"] *= age_of_universe

    def get_basic_quantities(self):
        """Calculates basic posterior quantities, these are fast as they
        are derived only from the SFH model, not the spectral model. """

        if "stellar_mass" in list(self.samples):
            return

        self.fitted_model._update_model_components(self.samples2d[0, :])
        self.sfh = star_formation_history(self.fitted_model.model_components)

        quantity_names = ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr",
                          "mass_weighted_age", "tform", "tquench",
                          "mass_weighted_metallicity"]

        for q in quantity_names:
            self.samples[q] = np.zeros(self.n_samples)

        self.samples["sfh"] = np.zeros((self.n_samples,
                                        self.sfh.ages.shape[0]))

        quantity_names += ["sfh"]

        for i in range(self.n_samples):
            param = self.samples2d[self.indices[i], :]
            self.fitted_model._update_model_components(param)
            self.sfh.update(self.fitted_model.model_components)

            for q in quantity_names:
                self.samples[q][i] = getattr(self.sfh, q)

    def get_advanced_quantities(self):
        """Calculates advanced derived posterior quantities, these are
        slower because they require the full model spectra. """

        if "spectrum_full" in list(self.samples):
            return

        self.fitted_model._update_model_components(self.samples2d[0, :])
        self.model_galaxy = model_galaxy(self.fitted_model.model_components,
                                         filt_list=self.galaxy.filt_list,
                                         spec_wavs=self.galaxy.spec_wavs,
                                         index_list=self.galaxy.index_list)

        all_names = ["photometry", "spectrum", "spectrum_full", "uvj",
                     "indices"]

        all_model_keys = dir(self.model_galaxy)
        quantity_names = [q for q in all_names if q in all_model_keys]

        for q in quantity_names:
            size = getattr(self.model_galaxy, q).shape[0]
            self.samples[q] = np.zeros((self.n_samples, size))

        if self.galaxy.photometry_exists:
            self.samples["chisq_phot"] = np.zeros(self.n_samples)

        if "dust" in list(self.fitted_model.model_components):
            size = self.model_galaxy.spectrum_full.shape[0]
            self.samples["dust_curve"] = np.zeros((self.n_samples, size))

        if "calib" in list(self.fitted_model.model_components):
            size = self.model_galaxy.spectrum.shape[0]
            self.samples["calib"] = np.zeros((self.n_samples, size))

        if "noise" in list(self.fitted_model.model_components):
            type = self.fitted_model.model_components["noise"]["type"]
            if type.startswith("GP"):
                size = self.model_galaxy.spectrum.shape[0]
                self.samples["noise"] = np.zeros((self.n_samples, size))

        for i in range(self.n_samples):
            param = self.samples2d[self.indices[i], :]
            self.fitted_model._update_model_components(param)
            self.fitted_model.lnlike(param)

            if self.galaxy.photometry_exists:
                self.samples["chisq_phot"][i] = self.fitted_model.chisq_phot

            if "dust" in list(self.fitted_model.model_components):
                dust_curve = self.fitted_model.model_galaxy.dust_atten.A_cont
                self.samples["dust_curve"][i] = dust_curve

            if "calib" in list(self.fitted_model.model_components):
                self.samples["calib"][i] = self.fitted_model.calib.model

            if "noise" in list(self.fitted_model.model_components):
                type = self.fitted_model.model_components["noise"]["type"]
                if type.startswith("GP"):
                    self.samples["noise"][i] = self.fitted_model.noise.mean()

            for q in quantity_names:
                if q == "spectrum":
                    spectrum = getattr(self.fitted_model.model_galaxy, q)[:, 1]
                    self.samples[q][i] = spectrum
                    continue

                self.samples[q][i] = getattr(self.fitted_model.model_galaxy, q)

    def predict(self, filt_list=None, spec_wavs=None, spec_units="ergscma",
                phot_units="ergscma", index_list=None):
        """Obtain posterior predictions for new observables not included
        in the data. """

        self.prediction = {}

        self.fitted_model._update_model_components(self.samples2d[0, :])
        model = model_galaxy(self.fitted_model.model_components,
                             filt_list=filt_list, phot_units=phot_units,
                             spec_wavs=spec_wavs, index_list=index_list)

        all_names = ["photometry", "spectrum", "indices"]

        all_model_keys = dir(model)
        quantity_names = [q for q in all_names if q in all_model_keys]

        for q in quantity_names:
            size = getattr(model, q).shape[0]
            self.prediction[q] = np.zeros((self.n_samples, size))

        for i in range(self.n_samples):
            param = self.samples2d[self.indices[i], :]
            self.fitted_model._update_model_components(param)
            model.update(self.fitted_model.model_components)

            for q in quantity_names:
                if q == "spectrum":
                    spectrum = getattr(model, q)[:, 1]
                    self.prediction[q][i] = spectrum
                    continue

                self.prediction[q][i] = getattr(model, q)

    def predict_basic_quantities_at_redshift(self, redshift,
                                             sfh_type="dblplaw"):
        """ Predicts basic (SFH-based) quantities at a specified higher
        redshift. This is a bit experimental, there's probably a better
        way. Only works for models with a single SFH component. """

        self.prediction_at_z = {}

        #if "stellar_mass" in list(self.prediction_at_z):
        #    return

        self.fitted_model._update_model_components(self.samples2d[0, :])
        self.sfh = star_formation_history(self.fitted_model.model_components)

        quantity_names = ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr",
                          "mass_weighted_age", "tform", "tquench"]

        for q in quantity_names:
            self.prediction_at_z[q] = np.zeros(self.n_samples)

        self.prediction_at_z["sfh"] = np.zeros((self.n_samples,
                                                self.sfh.ages.shape[0]))

        quantity_names += ["sfh"]

        for i in range(self.n_samples):
            param = self.samples2d[self.indices[i], :]
            self.fitted_model._update_model_components(param)
            self.sfh.update(self.fitted_model.model_components)

            formed_mass_at_z = self.sfh.massformed_at_redshift(redshift)

            model_comp = deepcopy(self.fitted_model.model_components)

            model_comp["redshift"] = redshift
            model_comp[sfh_type]["massformed"] = formed_mass_at_z

            self.sfh.update(model_comp)

            for q in quantity_names:
                self.prediction_at_z[q][i] = getattr(self.sfh, q)

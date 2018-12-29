from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os
import time
import pandas as pd

from subprocess import call

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    nproc = MPI.COMM_WORLD.Get_size()

except ImportError:
    rank = 0
    nproc = 1

from ..input.galaxy import galaxy
from ..fitting.fit import fit
from .. import utils
from .merge_catalogue import merge


class fit_catalogue(object):

    """ Fit a model to a catalogue of galaxies.

    Parameters
    ----------

    IDs : list
        A list of ID numbers for galaxies in the catalogue

    fit_instructions : dict
        A dictionary containing the details of the model to be fitted to
        the data.

    load_data : function
        Function which takes ID as an argument and returns the model
        spectrum and photometry. Spectrum should come first and be an
        array with a column of wavelengths in Angstroms, a column of
        fluxes in erg/s/cm^2/A and a column of flux errors in the same
        units. Photometry should come second and be an array with a
        column of fluxes in microjanskys and a column of flux errors
        in the same units.

    spectrum_exists : bool - optional
        If the objects do not have spectroscopic data set this to False.
        In this case, load_data should only return photometry.

    photometry_exists : bool - optional
        If the objects do not have photometric data set this to False.
        In this case, load_data should only return a spectrum.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    make_plots : bool - optional
        Whether to make output plots for each object.

    cat_filt_list : list - optional
        The filt_list, or list of filt_lists for the catalogue.

    vary_filt_list : bool - optional
        If True, changes the filter list for each object. When True,
        each entry in cat_filt_list is expected to be a different
        filt_list corresponding to each object in the catalogue.

    redshifts : list - optional
        List of values for the redshift for each object to be fixed to.

    redshift_sigma : float - optional
        If this is set, the redshift for each object will be assigned a
        Gaussian prior centred on the value in redshifts with this
        standard deviation. Hard limits will be placed at 3 sigma.

    analysis_function : function - optional
        Specify some function to be run on each completed fit, must
        take the fit object as its only argument.

    time_calls : bool - optional
        Whether to print information on the average time taken for
        likelihood calls.
    """

    def __init__(self, IDs, fit_instructions, load_data, spectrum_exists=True,
                 photometry_exists=True, make_plots=False, cat_filt_list=None,
                 vary_filt_list=False, redshifts=None, redshift_sigma=0.,
                 run=".", analysis_function=None, time_calls=False):

        self.IDs = np.array(IDs).astype(str)
        self.fit_instructions = fit_instructions
        self.load_data = load_data
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.make_plots = make_plots
        self.cat_filt_list = cat_filt_list
        self.vary_filt_list = vary_filt_list
        self.redshifts = redshifts
        self.redshift_sigma = redshift_sigma
        self.run = run
        self.analysis_function = analysis_function
        self.time_calls = time_calls

        self.n_objects = len(self.IDs)

        if rank == 0:
            utils.make_dirs()

            # Set up the directory for the output catalogues to be saved
            if not os.path.exists("pipes/cats/" + self.run):
                os.mkdir("pipes/cats/" + self.run)

            np.savetxt("pipes/cats/" + self.run + "/IDs", self.IDs, fmt="%s")

    def fit(self, verbose=False, n_live=400):
        """ Run through the catalogue, only fitting objects which have
        not already been started by another thread.

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get progress updates from the sampler.

        n_live : int - optional
            Number of live points: reducing speeds up the code but may
            lead to unreliable results.
        """
        if rank == 0:
            if os.path.exists("pipes/cats/" + self.run + "/kill"):
                call(["rm", "pipes/cats/" + self.run + "/kill"])

            n = 0

        for i in range(self.n_objects):

            if rank == 0:
                done = os.path.exists("pipes/cats/" + self.run + "/"
                                      + self.IDs[i] + ".lock")

                for proc in range(1, nproc):
                    MPI.COMM_WORLD.send(done, dest=proc)

            else:
                done = MPI.COMM_WORLD.recv(source=0)

            if done:
                continue

            self._fit_object(self.IDs[i], verbose=verbose, n_live=n_live)

            if rank == 0:
                if n == 10:
                    n = 0

                # Set up output catalogue
                if n == 0:
                    self.time0 = time.time()
                    outcat = self._setup_catalogue()

                outcat.loc[n, "#ID"] = self.galaxy.ID

                samples = self.fit.posterior.samples

                for v in self.vars:
                    outcat.loc[n, v + "_16"] = np.percentile(samples[v], 16)
                    outcat.loc[n, v + "_50"] = np.percentile(samples[v], 50)
                    outcat.loc[n, v + "_84"] = np.percentile(samples[v], 84)

                if self.redshifts is not None:
                    outcat.loc[n, "input_redshift"] = self.redshifts[i]

                outcat.loc[n, "log_evidence"] = self.fit.results["lnz"]
                outcat.loc[n, "log_evidence_err"] = self.fit.results["lnz_err"]

                # Check to see if the kill switch has been set
                if os.path.exists("pipes/cats/" + self.run + "/kill"):
                    sys.exit("Kill command received")

                # Save the updated output catalogue.
                outcat.to_csv("pipes/cats/" + self.run + "/" + self.run
                              + ".txt" + str(self.time0), sep="\t",
                              index=False)

                merge(self.run)

                n += 1

    def _set_redshift(self, ID):
        """ Sets the corrrect redshift (range) in self.fit_instructions
        for the object being fitted. """

        if self.redshifts is not None:
            ind = np.argmax(self.IDs == ID)

            if self.redshift_sigma > 0.:
                z = self.redshifts[ind]
                sig = self.redshift_sigma
                self.fit_instructions["redshift_prior_mu"] = z
                self.fit_instructions["redshift_prior_sigma"] = sig
                self.fit_instructions["redshift"] = (z - 3*sig, z + 3*sig)

            else:
                self.fit_instructions["redshift"] = self.redshifts[ind]

    def _fit_object(self, ID, verbose=False, n_live=400):
        """ Fit the specified object. """

        # Check to see if the kill switch has been set and if so stop.
        if os.path.exists("pipes/cats/" + self.run + "/kill"):
            sys.exit("Kill command received")

        # Save lock file to stop other threads from fitting this object
        if rank == 0:
            np.savetxt(utils.working_dir + "/pipes/cats/" + self.run
                       + "/" + str(ID) + ".lock", np.array([0.]))

        # Set the correct redshift for this object
        self._set_redshift(ID)

        # Get the correct filt_list for this object
        filt_list = self.cat_filt_list
        if self.vary_filt_list:
            filt_list = self.cat_filt_list[np.argmax(self.IDs == ID)]

        # Load up the observational data for this object
        self.galaxy = galaxy(ID, self.load_data, filt_list=filt_list,
                             spectrum_exists=self.spectrum_exists,
                             photometry_exists=self.photometry_exists)

        # Fit the object
        self.fit = fit(self.galaxy, self.fit_instructions, run=self.run,
                       time_calls=self.time_calls)

        self.fit.fit(verbose=verbose, n_live=n_live)

        if rank == 0:
            if self.analysis_function is not None:
                self.analysis_function(self.fit)

            # Make plots if necessary
            if self.make_plots:
                self.fit.plot_spectrum_posterior()
                self.fit.plot_corner()
                self.fit.plot_1d_posterior()
                self.fit.plot_sfh_posterior()

                if "calib" in list(self.fit.fitted_model.fit_instructions):
                    self.fit.plot_calibration()

    def _setup_catalogue(self):
        """ Set up and save the initial blank output catalogue. """

        self.vars = self.fit.fitted_model.params
        self.vars += ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr",
                      "mass_weighted_age", "tform", "tquench"]

        cols = ["#ID"]
        for var in self.vars:
            cols += [var + "_16", var + "_50", var + "_84"]

        cols += ["input_redshift", "log_evidence", "log_evidence_err"]

        outcat = pd.DataFrame(np.zeros((10, len(cols))), columns=cols)
        outcat.loc[:, "#ID"] = np.nan

        return outcat

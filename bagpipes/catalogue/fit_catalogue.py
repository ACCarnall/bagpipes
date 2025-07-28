from __future__ import print_function, division, absolute_import

import numpy as np
import os
import pandas as pd
import copy

from astropy.table import Table
from glob import glob

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    rank = 0
    size = 1

from ..input.galaxy import galaxy
from ..fitting.fit import fit
from .. import utils


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

    n_posterior : int - optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete for each object. Default 500.

    full_catalogue : bool - optional
        Adds minimum chi-squared values and rest-frame UVJ mags to the
        output catalogue, takes extra time, default False.

    load_data_kwargs : dict - optional
        Any additional keyword arguments to be passed to load_data.
    """

    def __init__(self, IDs, fit_instructions, load_data, spectrum_exists=True,
        photometry_exists=True, make_plots=False, cat_filt_list=None,
        vary_filt_list=False, redshifts=None, redshift_sigma=0.,
        run=".", analysis_function=None, time_calls=False,
        n_posterior=500, full_catalogue=False, load_indices=None,
        index_list=None, track_backlog=False, save_pdf_txts=True, 
        em_line_fluxes_to_save = ['Halpha', 'Hbeta', 'Hgamma', 'OIII_5007', 'OIII_4959', 'NII_6548', 'NII_6584'],
        em_line_ratios_to_save = ["OIII_4959+OIII_5007__Hbeta", "Halpha__Hbeta", "Hbeta__Hgamma", "NII_6548+NII_6584__Halpha"],
        load_data_kwargs = {},
    ):

        self.IDs = np.array(IDs).astype(str)
        if type(fit_instructions) is list:
            print('Fit instructions is a list. Setting individual. ')
            self.fit_instructions = fit_instructions[0]
            self.fit_instructions_list = fit_instructions
        else:
            self.fit_instructions = fit_instructions
            self.fit_instructions_list = None
        self.load_data = load_data
        self.load_data_kwargs = load_data_kwargs
        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.make_plots = make_plots
        self.cat_filt_list = cat_filt_list
        self.vary_filt_list = vary_filt_list
        self.redshifts = redshifts
        self.redshift_sigma = redshift_sigma
        self.run = run
        self.analysis_function = analysis_function
        self.save_pdf_txts = save_pdf_txts
        self.time_calls = time_calls
        self.n_posterior = n_posterior
        self.full_catalogue = full_catalogue
        self.load_indices = load_indices
        self.index_list = index_list
        self.em_line_fluxes_to_save = em_line_fluxes_to_save
        self.em_line_ratios_to_save = em_line_ratios_to_save

        self.n_objects = len(self.IDs)
        self.done = np.zeros(self.IDs.shape[0]).astype(bool)
        self.cat = None
        self.vars = None

        if rank == 0:
            utils.make_dirs(run=run)

    def fit(self, verbose=False, n_live=400, mpi_serial=False,
            track_backlog=False, sampler="multinest", pool=1, use_mpi=True, overwrite_h5=False):
        """ Run through the catalogue fitting each object.

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get progress updates from the sampler.

        n_live : int - optional
            Number of live points: reducing speeds up the code but may
            lead to unreliable results.

        mpi_serial : bool - optional
            When running through mpirun/mpiexec, the default behaviour
            is to fit one object at a time, using all available cores.
            When mpi_serial=True, each core will fit different objects.

        track_backlog : bool - optional
            When using mpi_serial, report the number of objects waiting
            to be added to the catalogue by the "zero" core that
            compiles results from all the others. High numbers mean
            cores are waiting around doing nothing.
        """

        if rank == 0:
            cat_file = "pipes/cats/" + self.run + ".fits"
            if os.path.exists(cat_file):
                self.cat = Table.read(cat_file).to_pandas()
                self.cat.index = self.IDs
                self.done = (self.cat.loc[:, "log_evidence"] != 0.).values

        if size > 1 and mpi_serial and use_mpi:
            print('Fitting with MPI_serial: 1 galaxy per core.')
            self._fit_mpi_serial(n_live=n_live, track_backlog=track_backlog)
            return

        for i in range(self.n_objects):

            # Check to see if the object has been fitted already
            if rank == 0:
                obj_done = self.done[i]

                for j in range(1, size):
                    comm.send(obj_done, dest=j)

            else:
                obj_done = comm.recv(source=0)

            if obj_done:
                continue

            # If not fit the object and update the output catalogue
            self._fit_object(self.IDs[i], verbose=verbose, n_live=n_live,
                             sampler=sampler, pool=pool, use_MPI=use_mpi, overwrite_h5=overwrite_h5)

            self.done[i] = True

            # Save the updated output catalogue.
            if rank == 0:
                save_cat = Table.from_pandas(self.cat)
                save_cat.write("pipes/cats/" + self.run + ".fits",
                               format="fits", overwrite=True)

                print("Bagpipes:", np.sum(self.done), "out of",
                      self.done.shape[0], "objects completed.")

    def _fit_mpi_serial(self, verbose=False, n_live=400,
                        track_backlog=False, sampler="multinest", overwrite_h5=False):
        """ Run through the catalogue fitting multiple objects at once
        on different cores. """

        self.done = self.done.astype(int)
        self.done[self.done == 1] += 1

        if rank == 0:  # The 0 process manages others, does no fitting
            print('Fitting multiple objects using MPI')
            for i in range(1, size):
                if not np.min(self.done):  # give out first IDs to fit
                    newID = self.IDs[np.argmin(self.done)]
                    comm.send(newID, dest=i)
                    self.done[np.argmin(self.done)] += 1

                else:  # Alternatively tell process all objects are done
                    comm.send(None, dest=i)

            if np.min(self.done) == 2:  # If all objects are done end
                return

            while True:  # Add results to catalogue + distribute new IDs
                # Wait for an object to be finished by any process
                oldID, done_rank = comm.recv(source=MPI.ANY_SOURCE)
                self.done[self.IDs == oldID] += 1  # mark as done

                if not np.min(self.done):  # Send new ID to process
                    newID = self.IDs[np.argmin(self.done != 0)]
                    self.done[self.IDs == newID] += 1   # mark in prep
                    comm.send(newID, dest=done_rank)   # send new ID

                else:  # Alternatively tell process all objects are done
                    comm.send(None, dest=done_rank)

                # Load posterior for finished object to update catalogue
                print('Rank 0 core generating posterior and updating catalogue for', oldID)
                self._fit_object(oldID, use_MPI=False, verbose=False,
                                 n_live=n_live, sampler=sampler, overwrite_h5=overwrite_h5)

                save_cat = Table.from_pandas(self.cat)
                save_cat.write("pipes/cats/" + self.run + ".fits",
                               format="fits", overwrite=True)

                if track_backlog:
                    n_done = len(glob("pipes/posterior/" + self.run + "/*.h5"))
                    n_cat = np.sum(self.cat["stellar_mass_50"] > 0.)
                    backlog = n_done - n_cat

                    print("Bagpipes:", np.sum(self.done == 2), "out of",
                          self.done.shape[0], "objects completed.",
                          "Backlog:", backlog, "/", size-1, "cores")
                else:
                    print("Bagpipes:", np.sum(self.done == 2), "out of",
                          self.done.shape[0], "objects completed.")

                if np.min(self.done) == 2:  # if all objects done end
                    return

        else:  # All ranks other than 0 fit objects as directed by 0
            while True:
                ID = comm.recv(source=0)  # receive new ID to fit

                if ID is None:  # If no new ID is given then end
                    return

                self.n_posterior = 5 # hacky, these don't get used
                self._fit_object(ID, use_MPI=False, verbose=False,
                                 n_live=n_live, sampler=sampler, overwrite_h5=overwrite_h5)

                comm.send([ID, rank], dest=0)  # Tell 0 object is done

    def _set_redshift(self, ID):
        """ Sets the corrrect redshift (range) in self.fit_instructions
        for the object being fitted. """

        if self.redshifts is not None:
            ind = np.argmax(self.IDs == ID)

            if self.redshift_sigma is not None:
                if isinstance(self.redshift_sigma, float):
                    if self.redshift_sigma > 0.:
                        z = self.redshifts[ind]
                        sig = self.redshift_sigma
                        self.fit_instructions["redshift_prior"] = "Gaussian"
                        self.fit_instructions["redshift_prior_mu"] = z
                        self.fit_instructions["redshift_prior_sigma"] = sig
                        self.fit_instructions["redshift"] = (z - 3*sig, z + 3*sig)
                    else:
                        self.fit_instructions["redshift"] = self.redshifts[ind]

                elif (isinstance(self.redshift_sigma, np.ndarray) | isinstance(self.redshift_sigma, list)) & (self.redshift_sigma[ind] > 0.):
                    z = self.redshifts[ind]
                    sig = self.redshift_sigma[ind]
                    self.fit_instructions["redshift_prior_mu"] = z
                    self.fit_instructions["redshift_prior_sigma"] = sig
                    self.fit_instructions["redshift"] = (z - 3*sig, z + 3*sig)
                else:
                    self.fit_instructions["redshift"] = self.redshifts[ind]
            else:
                self.fit_instructions["redshift"] = self.redshifts[ind]

    def _fit_object(self, ID, verbose=False, n_live=400, use_MPI=True,
                    sampler="multinest", pool=1, overwrite_h5=False):
        """ Fit the specified object and update the catalogue. """

        if self.fit_instructions_list is not None:
            print('Setting fit_instructions from list.')
            self.fit_instructions = self.fit_instructions_list[np.argmax(self.IDs == ID)]
            
        # Set the correct redshift for this object
        self._set_redshift(ID)

        # Get the correct filt_list for this object
        filt_list = self.cat_filt_list
        if self.vary_filt_list:
            filt_list = self.cat_filt_list[np.argmax(self.IDs == ID)]

        # Load up the observational data for this object
        self.galaxy = galaxy(ID, self.load_data, filt_list=filt_list,
                             spectrum_exists=self.spectrum_exists,
                             photometry_exists=self.photometry_exists,
                             load_indices=self.load_indices,
                             index_list=self.index_list, 
                             em_line_fluxes_to_save = self.em_line_fluxes_to_save,
                             em_line_ratios_to_save = self.em_line_ratios_to_save,
                             load_data_kwargs=self.load_data_kwargs)

        # Fit the object
        self.obj_fit = fit(self.galaxy, self.fit_instructions, run=self.run,
                           time_calls=self.time_calls,
                           n_posterior=self.n_posterior)

        self.obj_fit.fit(verbose=verbose, n_live=n_live, use_MPI=use_MPI,
                         sampler=sampler, pool=pool, overwrite_h5 = overwrite_h5)

        if rank == 0 or not use_MPI:
            if self.vars is None:
                self._setup_vars()

            if self.cat is None:
                self._setup_catalogue()

            if self.analysis_function is not None:
                self.analysis_function(self.obj_fit)

            # Make plots if necessary - wrap in try/except
            if self.make_plots:
                plots = [
                    self.obj_fit.plot_spectrum_posterior,
                    self.obj_fit.plot_corner,
                    self.obj_fit.plot_1d_posterior,
                    self.obj_fit.plot_sfh_posterior,
                    self.obj_fit.plot_csfh_posterior
                ]
                names = [
                    "spectrum_posterior",
                    "corner",
                    "1d_posterior",
                    "sfh_posterior",
                    "csfh_posterior"
                ]
                if "calib" in list(self.obj_fit.fitted_model.fit_instructions):
                    plots.append(self.obj_fit.plot_calibration)
                    names.append("calibration")
                for plot, name in zip(plots, names):
                    try:
                        plot()
                    except:
                        print(f"Error plotting {name} for {ID}")

            # Add fitting results to output catalogue
            if self.full_catalogue:
                self.obj_fit.posterior.get_advanced_quantities()
                
            if size > 1:
                # This little hack is needed as the individual cores only generate 5 samples
                self.obj_fit.add_quantities_to_h5()

            samples = self.obj_fit.posterior.samples

            for v in self.vars:

                if v == "UV_colour":
                    values = samples["uvj"][:, 0] - samples["uvj"][:, 1]

                elif v == "VJ_colour":
                    values = samples["uvj"][:, 1] - samples["uvj"][:, 2]

                else:
                    values = samples[v]

                if self.save_pdf_txts:
                    self._save_PDF(v, values, ID)
                
                self.cat.loc[ID, v + "_16"] = np.percentile(values, 16)
                self.cat.loc[ID, v + "_50"] = np.percentile(values, 50)
                self.cat.loc[ID, v + "_84"] = np.percentile(values, 84)

            results = self.obj_fit.results
            self.cat.loc[ID, "log_evidence"] = results["lnz"]
            self.cat.loc[ID, "log_evidence_err"] = results["lnz_err"]

            if self.full_catalogue and self.photometry_exists:
                self.cat.loc[ID, "chisq_phot"] = np.min(samples["chisq_phot"])
                n_bands = np.sum(self.galaxy.photometry[:, 1] != 0.)
                self.cat.loc[ID, "n_bands"] = n_bands
    
    def _save_PDF(self, var_name, values, ID):
        pdf_file = f"pipes/pdfs/{self.run}/{var_name}/{ID}.txt"
        os.makedirs("/".join(pdf_file.split("/")[:-1]), exist_ok = True)
        os.chmod("/".join(pdf_file.split("/")[:-1]), 0o777)
        np.savetxt(pdf_file, values)
        os.chmod(pdf_file, 0o777)
    
    def _setup_vars(self):
        """ Set up list of variables to go in the output catalogue. """

        self.vars = copy.copy(self.obj_fit.fitted_model.params)
        self.vars += ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr",
                          "sfr_10myr","ssfr_10myr", "nsfr_10myr", "burstiness",
                          "mass_weighted_age", "tform", "tquench",
                          "mass_weighted_zmet"]

        if self.full_catalogue:
            self.vars += ["UV_colour", "VJ_colour"]
            self.vars += ["beta_C94", "m_UV", "M_UV", "D4000"]
            for frame in ["rest", "obs"]:
                for property in ["xi_ion_caseB", "ndot_ion_caseB"]:
                    self.vars += [f"{property}_{frame}"]
                for line in self.em_line_fluxes_to_save:
                    self.vars += [f"{line}_flux_{frame}", f"{line}_EW_{frame}"]
            for ratio in self.em_line_ratios_to_save:
                self.vars += [ratio]
            for line in self.em_line_fluxes_to_save:
                self.vars += [f"{line}_cont"]

    def _setup_catalogue(self):
        """ Set up the initial blank output catalogue. """

        cols = ["#ID"]

        for var in self.vars:
            cols += [var + "_16", var + "_50", var + "_84"]

        cols += ["input_redshift", "log_evidence", "log_evidence_err"]

        if self.full_catalogue and self.photometry_exists:
            cols += ["chisq_phot", "n_bands"]

        self.cat = pd.DataFrame(np.zeros((self.IDs.shape[0], len(cols))), columns=cols)

        self.cat.loc[:, "#ID"] = self.IDs
        self.cat.index = self.IDs

        if self.redshifts is not None:
            self.cat.loc[:, "input_redshift"] = self.redshifts

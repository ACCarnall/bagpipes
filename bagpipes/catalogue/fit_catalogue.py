from __future__ import print_function, division, absolute_import

import numpy as np
import os
import sys
import time
import pandas as pd
import copy
import traceback

from astropy.table import Table
from astropy.io import fits
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

    redshift_sigma : float or array-like - optional
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
    """

    def __init__(self, IDs, fit_instructions, load_data, spectrum_exists=True,
                 photometry_exists=True, make_plots=False, cat_filt_list=None,
                 vary_filt_list=False, redshifts=None, redshift_sigma=None,
                 run=".", analysis_function=None, time_calls=False,
                 n_posterior=500, full_catalogue=False, load_indices=None,
                 index_list=None, track_backlog=False, spec_units="ergscma",
                 phot_units="mujy"):

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
        self.n_posterior = n_posterior
        self.full_catalogue = full_catalogue
        self.load_indices = load_indices
        self.index_list = index_list
        self.spec_units = spec_units
        self.phot_units = phot_units

        self.n_objects = len(self.IDs)
        self.done = np.zeros(self.IDs.shape[0]).astype(bool)
        self.cat = None
        self.vars = None

        # Layout of the saved catalogue file, set once it has been written
        # in full, after which rows are updated in place (see _save_catalogue).
        self._cat_layout = None

        # IDs of objects which failed to fit (when using mpi_serial).
        self.failed = []

        if rank == 0:
            utils.make_dirs(run=run)

    def fit(self, verbose=False, n_live=400, mpi_serial=False,
            track_backlog=False, sampler="multinest", pool=1):
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

        if size == 1:
            self._fit(verbose=verbose, n_live=n_live, mpi_serial=mpi_serial,
                      track_backlog=track_backlog, sampler=sampler, pool=pool)
            return

        # Under MPI, an unhandled error on one process would otherwise
        # leave the others waiting for it forever, as Python's exit
        # handler blocks in MPI_Finalize. Abort the whole job instead.
        try:
            self._fit(verbose=verbose, n_live=n_live, mpi_serial=mpi_serial,
                      track_backlog=track_backlog, sampler=sampler, pool=pool)

        except BaseException:
            traceback.print_exc()
            print("Bagpipes: unhandled error on MPI process " + str(rank)
                  + ", aborting all processes.", flush=True)
            sys.stderr.flush()

            # Give mpirun time to forward the error output before the
            # abort kills all processes, otherwise it can be lost.
            time.sleep(2)
            comm.Abort(1)

    def _fit(self, verbose=False, n_live=400, mpi_serial=False,
             track_backlog=False, sampler="multinest", pool=1):
        """ Run through the catalogue fitting each object (see fit). """

        if rank == 0:
            cat_file = "pipes/cats/" + self.run + ".fits"
            if os.path.exists(cat_file):
                self.cat = Table.read(cat_file).to_pandas()
                self.cat.index = self.IDs
                self.done = (self.cat.loc[:, "log_evidence"] != 0.).values

        if size > 1 and mpi_serial:
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
                             sampler=sampler, pool=pool)

            self.done[i] = True

            # Save the updated output catalogue.
            if rank == 0:
                self._save_catalogue(self.IDs[i])

                print("Bagpipes:", np.sum(self.done), "out of",
                      self.done.shape[0], "objects completed.")

    def _fit_mpi_serial(self, verbose=False, n_live=400,
                        track_backlog=False, sampler="multinest"):
        """ Run through the catalogue fitting multiple objects at once
        on different cores. """

        self.done = self.done.astype(int)
        self.done[self.done == 1] += 1

        if rank == 0:  # The 0 process manages others, does no fitting
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
                # Wait for an object to be finished by any process, which
                # also sends the catalogue row it has computed, or the
                # error message if fitting the object failed.
                msg = comm.recv(source=MPI.ANY_SOURCE)
                oldID, done_rank, cat_vars, row, error = msg
                self.done[self.IDs == oldID] += 1  # mark as done

                if not np.min(self.done):  # Send new ID to process
                    newID = self.IDs[np.argmin(self.done != 0)]
                    self.done[self.IDs == newID] += 1   # mark in prep
                    comm.send(newID, dest=done_rank)   # send new ID

                else:  # Alternatively tell process all objects are done
                    comm.send(None, dest=done_rank)

                # Add the results for the finished object to the catalogue
                if error is None:
                    self._add_to_catalogue(oldID, cat_vars, row)
                    self._save_catalogue(oldID)

                # The object failed: leave its catalogue row empty, so that
                # it will be tried again if fit_catalogue is re-run.
                else:
                    self.failed.append(str(oldID))
                    print("Bagpipes: fitting object " + str(oldID)
                          + " failed on process " + str(done_rank) + ": "
                          + error.strip().split("\n")[-1])

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
                    if self.failed:
                        print("Bagpipes:", len(self.failed), "objects failed,"
                              + " re-run to try them again:", self.failed)
                    return

        else:  # All ranks other than 0 fit objects as directed by 0
            while True:
                ID = comm.recv(source=0)  # receive new ID to fit

                if ID is None:  # If no new ID is given then end
                    return

                # If something goes wrong with this object, report it to
                # process 0 and move on, rather than stopping this process
                # (which would leave process 0 waiting for it forever).
                try:
                    self._run_fit(ID, use_MPI=False, verbose=False,
                                  n_live=n_live, sampler=sampler)

                    # Compute the catalogue row here, so that process 0
                    # does not have to reload the posterior to do this.
                    row = self._get_catalogue_row(ID)
                    msg = [ID, rank, self.vars, row, None]

                except Exception:
                    error = traceback.format_exc()
                    print("Bagpipes: fitting object " + str(ID) + " failed"
                          + " on process " + str(rank) + ":\n" + error,
                          flush=True)

                    msg = [ID, rank, None, None, error]

                # Tell 0 object is done and send the results
                comm.send(msg, dest=0)

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

                elif (isinstance(self.redshift_sigma, (list, np.ndarray))
                      & (self.redshift_sigma[ind] > 0.)):
                    z = self.redshifts[ind]
                    sig = self.redshift_sigma[ind]
                    self.fit_instructions["redshift_prior"] = "Gaussian"
                    self.fit_instructions["redshift_prior_mu"] = z
                    self.fit_instructions["redshift_prior_sigma"] = sig
                    self.fit_instructions["redshift"] = (z - 3*sig, z + 3*sig)
                else:
                    self.fit_instructions["redshift"] = self.redshifts[ind]

            else:
                self.fit_instructions["redshift"] = self.redshifts[ind]

    def _fit_object(self, ID, verbose=False, n_live=400, use_MPI=True,
                    sampler="multinest", pool=1):
        """ Fit the specified object and update the catalogue. """

        self._run_fit(ID, verbose=verbose, n_live=n_live, use_MPI=use_MPI,
                      sampler=sampler, pool=pool)

        if rank == 0:
            row = self._get_catalogue_row(ID)
            self._add_to_catalogue(ID, self.vars, row)

    def _run_fit(self, ID, verbose=False, n_live=400, use_MPI=True,
                 sampler="multinest", pool=1):
        """ Fit the specified object (or load the results if it has
        already been fitted). """

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
                             spec_units=self.spec_units,
                             phot_units=self.phot_units)

        # Fit the object
        self.obj_fit = fit(self.galaxy, self.fit_instructions, run=self.run,
                           time_calls=self.time_calls,
                           n_posterior=self.n_posterior)

        self.obj_fit.fit(verbose=verbose, n_live=n_live, use_MPI=use_MPI,
                         sampler=sampler, pool=pool)

    def _get_catalogue_row(self, ID):
        """ Run any analysis function and plotting for the most recently
        fitted object, and return its row of the output catalogue as a
        dictionary of column names and values. """

        if self.vars is None:
            self._setup_vars()

        if self.analysis_function is not None:
            self.analysis_function(self.obj_fit)

        # Make plots if necessary
        if self.make_plots:
            self.obj_fit.plot_spectrum_posterior()
            self.obj_fit.plot_corner()
            self.obj_fit.plot_1d_posterior()
            self.obj_fit.plot_sfh_posterior()

            if "calib" in list(self.obj_fit.fitted_model.fit_instructions):
                self.obj_fit.plot_calibration()

        # Calculate fitting results for the output catalogue
        if self.full_catalogue:
            self.obj_fit.posterior.get_advanced_quantities()

        samples = self.obj_fit.posterior.samples

        row = {}

        for v in self.vars:

            if v == "UV_colour":
                values = samples["uvj"][:, 0] - samples["uvj"][:, 1]

            elif v == "VJ_colour":
                values = samples["uvj"][:, 1] - samples["uvj"][:, 2]

            else:
                values = samples[v]

            row[v + "_16"] = np.percentile(values, 16)
            row[v + "_50"] = np.percentile(values, 50)
            row[v + "_84"] = np.percentile(values, 84)

        results = self.obj_fit.results
        row["log_evidence"] = results["lnz"]
        row["log_evidence_err"] = results["lnz_err"]

        if self.full_catalogue and self.photometry_exists:
            row["chisq_phot"] = np.min(samples["chisq_phot"])
            row["n_bands"] = np.sum(self.galaxy.photometry[:, 1] != 0.)

        return row

    def _add_to_catalogue(self, ID, cat_vars, row):
        """ Add a row of results (from _get_catalogue_row) for the object
        ID to the output catalogue, setting the catalogue up if needed. """

        if self.vars is None:
            self.vars = cat_vars

        if self.cat is None:
            self._setup_catalogue()

        for col, value in row.items():
            self.cat.loc[ID, col] = value

    def _save_catalogue(self, ID):
        """ Save the output catalogue after the row for object ID has been
        updated. The first save in each run writes the whole file, after
        which only the updated row is written, in place. FITS binary
        tables have fixed-width rows, so this is much faster than writing
        the whole file every time, especially for large catalogues. """

        cat_file = "pipes/cats/" + self.run + ".fits"

        if self._cat_layout is None or not os.path.exists(cat_file):
            self._write_full_catalogue(cat_file)
            return

        data_start, row_dtype = self._cat_layout

        # Build the row in the on-disk format and write it at its offset.
        row_ind = self.cat.index.get_loc(ID)
        values = self.cat.iloc[row_ind].tolist()
        row = np.array([tuple(values)], dtype=row_dtype)

        offset = data_start + row_ind*row_dtype.itemsize

        with open(cat_file, "r+b") as f:
            f.seek(offset)
            f.write(row.tobytes())

    def _write_full_catalogue(self, cat_file):
        """ Write the whole output catalogue, then record the file layout
        so that later saves can update rows in place. The file is written
        to a temporary name then renamed, so an interrupted write cannot
        leave a corrupted or missing catalogue. """

        tmp_file = cat_file + ".tmp"
        Table.from_pandas(self.cat).write(tmp_file, format="fits",
                                          overwrite=True)
        os.replace(tmp_file, cat_file)

        self._cat_layout = None

        with fits.open(cat_file) as hdul:
            hdu = hdul[1]

            # Only update in place if the stored bytes are the values
            # themselves (no scaled/boolean columns or checksums), and the
            # columns are the same as in the catalogue.
            simple = all(c.bscale is None and c.bzero is None
                         and c.format[-1] != "L" for c in hdu.columns)
            simple &= not any(k in hdu.header for k in ("CHECKSUM",
                                                         "DATASUM"))
            simple &= list(hdu.columns.names) == list(self.cat.columns)
            simple &= hdu.header["NAXIS2"] == self.cat.shape[0]

            if simple:
                # FITS stores numbers big-endian, columns.dtype is native.
                self._cat_layout = (hdu.fileinfo()["datLoc"],
                                    hdu.columns.dtype.newbyteorder(">"))

    def _setup_vars(self):
        """ Set up list of variables to go in the output catalogue. """

        self.vars = copy.copy(self.obj_fit.fitted_model.params)
        self.vars += ["stellar_mass", "formed_mass", "sfr", "ssfr", "nsfr",
                      "mass_weighted_age", "tform", "tquench"]

        if self.full_catalogue:
            self.vars += ["UV_colour", "VJ_colour"]

    def _setup_catalogue(self):
        """ Set up the initial blank output catalogue. """

        cols = []
        for var in self.vars:
            cols += [var + "_16", var + "_50", var + "_84"]

        cols += ["input_redshift", "log_evidence", "log_evidence_err"]

        if self.full_catalogue and self.photometry_exists:
            cols += ["chisq_phot", "n_bands"]

        self.cat = pd.DataFrame(np.zeros((self.IDs.shape[0], len(cols))),
                                columns=cols)

        self.cat.loc[:, "#ID"] = self.IDs
        self.cat = self.cat[["#ID"] + cols]
        self.cat.index = self.IDs

        if self.redshifts is not None:
            self.cat.loc[:, "input_redshift"] = self.redshifts

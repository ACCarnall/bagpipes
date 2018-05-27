from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os

import glob
from astropy.io import fits
from subprocess import call

from . import utils
from .galaxy import galaxy
from .fit import fit


class catalogue_fit:

    """ Fit a model to a catalogue of galaxies.

    Parameters
    ----------

    Catalogue_IDs : list
        A list of ID numbers for galaxies in the catalogue

    fit_instructions : dict
        A dictionary containing the details of the model to be fitted to
        the data.

    data_load_function : function
        Function which takes ID, filtlist as its two arguments and
        returns the model spectrum and photometry. Spectrum should come
        first and be an array with a column of wavelengths in Angstroms,
        a column of fluxes in erg/s/cm^2/A and a column of flux errors
        in the same units. Photometry should come second and be an array
        with a column of fluxes in microjanskys and a column of flux
        errors in the same units.

    catalogue_filtlist : string or list (optional)
        The filtlist for the catalogue. Either a string, or a list of
        the same length as Catalogue_IDs

    spectrum_exists : bool(optional)
        If the objects do not have spectral data, set this to False.
        In this case, data_load_function should only return photometry.

    photometry_exists : bool (optional)
        If the objects do not have photometric data for, set this to
        False. In this case, data_load_function should only return a
        spectrum.

    run : string (optional)
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    catalogue_redshifts : list (optional)
        A list of redshift values associated with the objects to be
        fitted. Same length as Catalogue_IDs.

    fix_redshifts : bool or float (optional)
        Whether to fix the redshifts to the values specified in
        catalogue_redshifts, if True, redshifts are fixed, if a float,
        the redshifts will be allowed to vary within this range centred
        on the value given in catalogue_redshifts for that object.

    make_plots : bool (optional)
        If True, spectral and corner plots will be made for each object.

    """

    def __init__(self, catalogue_IDs, fit_instructions, data_load_func,
                 catalogue_filtlist=None, spectrum_exists=True,
                 photometry_exists=True, run=".", catalogue_redshifts=None,
                 fix_redshifts=False, save_phot=False, make_plots=False):

        self.run = run
        self.IDs = catalogue_IDs
        self.save_phot = save_phot
        self.make_plots = make_plots
        self.fix_redshifts = fix_redshifts
        self.data_load_func = data_load_func
        self.redshifts = catalogue_redshifts
        self.spectrum_exists = spectrum_exists
        self.fit_instructions = fit_instructions
        self.photometry_exists = photometry_exists
        self.catalogue_filtlist = catalogue_filtlist

        self.n_objects = catalogue_IDs.shape[0]

        make_dirs()

        if not os.path.exists(working_dir + "/pipes/cats/" + self.run):
            os.mkdir(working_dir + "/pipes/cats/" + self.run)

        np.savetxt(working_dir + "/pipes/cats/" + self.run + "/all_IDs",
                   self.IDs)

    def fit(self, verbose=False, n_live=400, sampler="dynesty"):
        """ Run through the catalogue, only fitting objects which have
        not already been started by another thread. """

        if os.path.exists(working_dir + "/pipes/cats/" + self.run + "/kill"):
            call(["rm", working_dir + "/pipes/cats/" + self.run + "/kill"])

        done = 0
        time0 = time.time()

        while True:

            for i in range(self.n_objects):

                if self.fix_redshifts:
                    self.fit_instructions["redshift"] = self.redshifts[i]

                elif isinstance(self.fix_redshifts, float):
                    z_range = (self.redshifts[i]-self.fix_redshifts/2.,
                               self.redshifts[i]+self.fix_redshifts/2.)

                    fit_instructions["redshift"] = z_range

                if not os.path.exists(working_dir + "/pipes/cats/" + self.run
                                      + "/" + str(int(self.IDs[i])) + ".lock"):

                    if os.path.exists(working_dir + "/pipes/cats/"
                                      + self.run + "/kill"):

                        sys.exit("Kill command received")

                    np.savetxt(working_dir + "/pipes/cats/" + self.run + "/"
                               + str(int(self.IDs[i])) + ".lock",
                               np.array([0.]))

                    # Extract filtlist for this object
                    if isinstance(self.catalogue_filtlist, (type(None), str)):
                        filtlist = self.catalogue_filtlist

                    else:
                        filtlist = self.catalogue_filtlist[i]

                    galaxy = Galaxy(str(int(self.IDs[i])), self.data_load_func,
                                    filtlist=filtlist,
                                    spectrum_exists=self.spectrum_exists,
                                    photometry_exists=self.photometry_exists)

                    fit = Fit(galaxy, self.fit_instructions, run=self.run)

                    """ Fit the object and make plots of the fit. """
                    fit.fit(verbose=verbose, n_live=n_live, sampler=sampler)

                    if self.make_plots:
                        fit.plot_fit()
                        fit.plot_corner()
                        fit.plot_1d_posterior()

                        if "polynomial" in self.fit_instructions.keys():
                            fit.plot_poly()

                    if done == 10:
                        done = 0
                        time0 = time.time()

                    """ Take care of output catalogue(s) """
                    if done == 0:
                        len_outcat = 3*fit.ndim + 20
                        outcat = np.zeros((10, len_outcat))
                        np.savetxt(working_dir + "/pipes/cats/" + self.run
                                   + "/" + self.run + ".txt" + str(time0),
                                   outcat)

                        if self.save_phot:
                            phot_bands = galaxy.photometry.shape[0]
                            photcat = np.zeros((10, 1 + 2*phot_bands))
                            np.savetxt(working_dir + "/pipes/cats/" + self.run
                                       + "/" + self.run + "_phot.txt"
                                       + str(time0), photcat)

                    outcat[done, 0] = fit.Galaxy.ID

                    post = fit.posterior

                    outcat[done, -19] = np.percentile(post["UVJ"][:, 0]
                                                      - post["UVJ"][:, 1], 16)
                    outcat[done, -18] = np.percentile(post["UVJ"][:, 0]
                                                      - post["UVJ"][:, 1], 50)
                    outcat[done, -17] = np.percentile(post["UVJ"][:, 0]
                                                      - post["UVJ"][:, 1], 84)

                    outcat[done, -16] = np.percentile(post["UVJ"][:, 1]
                                                      - post["UVJ"][:, 2], 16)

                    outcat[done, -15] = np.percentile(post["UVJ"][:, 1]
                                                      - post["UVJ"][:, 2], 50)

                    outcat[done, -14] = np.percentile(post["UVJ"][:, 1]
                                                      - post["UVJ"][:, 2], 84)

                    outcat[done, -13] = np.percentile(post["tmw"], 16)
                    outcat[done, -12] = np.percentile(post["tmw"], 50)
                    outcat[done, -11] = np.percentile(post["tmw"], 84)

                    outcat[done, -10] = np.percentile(post["sfr"], 16)
                    outcat[done, -9] = np.percentile(post["sfr"], 50)
                    outcat[done, -8] = np.percentile(post["sfr"], 84)

                    living_mass = post["mass"]["total"]["living"]
                    outcat[done, -7] = np.percentile(living_mass, 16)
                    outcat[done, -6] = np.percentile(living_mass, 50)
                    outcat[done, -5] = np.percentile(living_mass, 84)

                    if self.redshifts is not None:
                        outcat[done, -4] = self.redshifts[i]

                    outcat[done, -3] = fit.global_log_evidence
                    outcat[done, -2] = fit.global_log_evidence_err
                    outcat[done, -1] = fit.min_chisq_red

                    for j in range(fit.ndim):
                        outcat[done, 1 + 3*j] = fit.conf_int[j][0]
                        outcat[done, 2 + 3*j] = post_median[j]
                        outcat[done, 3 + 3*j] = fit.conf_int[j][1]

                    if self.save_phot:
                        photcat[done, 0] = fit.Galaxy.ID
                        photcat[done, 1:1+phot_bands] = galaxy.photometry[:, 1]
                        median_phot = np.percentile(post["photometry"],
                                                    50, axis=1)
                        photcat[done, 1+phot_bands:] = median_phot

                    """ Set up the header for the output catalogue. """
                    fileheader = "ID "

                    for j in range(fit.ndim):
                        fileheader += (fit.fit_params[j] + "_16 "
                                       + fit.fit_params[j] + "_median "
                                       + fit.fit_params[j] + "_84 ")

                    fileheader += ("UVcolour_16 UVcolour_median UVcolour_84"
                                   + " VJcolour_16 VJcolour_median VJcolour_84"
                                   + " tmw_16 tmw_median tmw_84 sfr_16"
                                   + " sfr_median sfr_84 mstar_liv_16"
                                   + " mstar_liv_median mstar_liv_84 z_input"
                                   + " global_log_evidence"
                                   + " global_log_evidence_err"
                                   + " min_chisq_reduced")

                    """ Check to see if the kill switch has been set,
                    and if so stop here. """
                    if os.path.exists(working_dir + "/pipes/cats/"
                                      + self.run + "/kill"):

                        sys.exit("Kill command received")

                    """ If not, save the updated output catalogue. """
                    np.savetxt(working_dir + "/pipes/cats/" + self.run + "/"
                               + self.run + ".txt" + str(time0), outcat,
                               header=fileheader)

                    if self.save_phot:
                        np.savetxt(working_dir + "/pipes/cats/" + self.run
                                   + "/" + self.run + "_phot.txt" + str(time0),
                                   photcat)

                    done += 1

                    merge_cat(self.run)

            else:

                break


def merge_cat(run, mode="merge"):
    """ Compile all the sub-catalogues into one output catalogue,
    optionally stop all running processes and delete incomplete object
    posteriors. """

    if mode == "clean":
        call(["touch", working_dir + "/pipes/cats/" + run + "/kill"])

    outcats = {}
    outphotcats = {}

    # Generate lists of files to merge
    files = glob.glob(working_dir + "/pipes/cats/" + run + "/" + run + ".txt*")

    photfiles = glob.glob(working_dir + "/pipes/cats/" + run + "/"
                          + run + "_phot.txt*")

    if len(photfiles) != 0:
        merge_phot = True

    else:
        merge_phot = False

    # Load up files
    for i in range(len(files)):
        while True:
            try:
                outcats[str(i)] = np.loadtxt(files[i])
                break

            except ValueError:
                time.sleep(1)

    if merge_phot:
        for i in range(len(photfiles)):
            while True:
                try:
                    outphotcats[str(i)] = np.loadtxt(photfiles[i])
                    break

                except ValueError:
                    time.sleep(1)

    # Generate files to merge outputs into
    f = open(files[0])
    header = f.readline()
    header_list = header[2:-1].split()
    f.close()

    all_IDs = np.loadtxt(working_dir + "/pipes/cats/" + run + "/all_IDs")

    outcat_final = np.zeros((all_IDs.shape[0], outcats["0"].shape[1]))
    outcat_final[:, 0] = all_IDs

    if merge_phot:
        outphotcat_final = np.zeros((all_IDs.shape[0],
                                     outphotcats["0"].shape[1]))
        outphotcat_final[:, 0] = all_IDs

    # Count the number of objects which have been fit by each thread
    n_in_thread = {}

    for k in range(len(files)):
        n_in_thread[str(k)] = 0

    # Merge output files
    for j in range(len(files)):
        mask = ((outcats[str(j)][:, 1] != 0.)
                & (outcats[str(j)][:, 2] != 0.))

        for k in range(outcats[str(j)][mask].shape[0]):
            mask = (outcat_final[:, 0] == outcats[str(j)][k, 0])
            outcat_final[mask, :] = outcats[str(j)][k, :]
            n_in_thread[str(j)] += 1

    if merge_phot:
        for j in range(len(photfiles)):
            mask = ((outphotcats[str(j)][:, 1] != 0.)
                    & (outphotcats[str(j)][:, 2] != 0.))

            for k in range(outphotcats[str(j)][mask].shape[0]):
                mask = (outphotcat_final[:, 0] == outphotcats[str(j)][k, 0])
                outphotcat_final[mask, :] = outphotcats[str(j)][k, :]

    # Count the number of objects finished
    nobj = 0

    for i in range(len(files)):
        nobj += n_in_thread[str(i)]

    # If mode is clean, remove all input catalogues and replace with a
    # merged one, also delete objects which are in progress
    if mode == "clean":
        for file in files:
            call(["rm",  file])

        if merge_phot:
            for photfile in photfiles:
                call(["rm",  photfile])

        mask = (outcat_final[:, 1] != 0.) & (outcat_final[:, 2] != 0.)
        np.savetxt(working_dir + "/pipes/cats/" + run + "/" + run
                   + ".txt_clean", outcat_final[mask], header=header[2:-1])

        if merge_phot:
            mask = ((outphotcat_final[:, 1] != 0.)
                    & (outphotcat_final[:, 2] != 0.))

            np.savetxt(working_dir + "/pipes/cats/" + run + "/" + run
                       + "_phot.txt_clean", outphotcat_final[mask])

        os.chdir("pipes/cats/" + run)
        lock_files = glob.glob("*.lock")
        os.chdir(working_dir)

        IDs = np.loadtxt(working_dir + "/pipes/cats/" + run + "/" + run
                         + ".txt_clean", usecols=(0, 1, 2))

        IDs = IDs[(IDs[:, 1] != 0.) & (IDs[:, 2] != 0.), 0]

        for lock_file in lock_files:
            if float(lock_file[:-5]) not in IDs:

                call(["rm", working_dir + "/pipes/cats/" + run + "/"
                      + lock_file])

                files_toremove = glob.glob(working_dir + "/pipes/posterior/"
                                           + run + "/" + lock_file[:-5] + "*")

                for toremove in files_toremove:
                    call(["rm", toremove])

    mask = (outcat_final[:, 1] != 0.) & (outcat_final[:, 2] != 0.)
    np.savetxt(working_dir + "/pipes/cats/" + run + ".cat",
               outcat_final[mask], header=header[2: -1])

    if merge_phot:
        mask = ((outphotcat_final[:, 1] != 0.)
                & (outphotcat_final[:, 2] != 0.))

        np.savetxt(working_dir + "/pipes/cats/" + run + "_phot.cat",
                   outphotcat_final[mask])

    print("Bagpipes:", nobj, "out of", outcat_final.shape[0],
          "objects completed.")

    if mode == "clean":
        print("Bagpipes: Running processes will be killed and partially"
              + " completed objects reset.")


def clean_cat(run):
    """ Run compile_cat with the clean option enabled to kill running
    processes and delete . """
    merge_cat(run, mode="clean")

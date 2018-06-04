from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os
import time

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

    load_data : function
        Function which takes ID, filt_list as its two arguments and
        returns the model spectrum and photometry. Spectrum should come
        first and be an array with a column of wavelengths in Angstroms,
        a column of fluxes in erg/s/cm^2/A and a column of flux errors
        in the same units. Photometry should come second and be an array
        with a column of fluxes in microjanskys and a column of flux
        errors in the same units.

    filt_list : list (optional)
        The filt_list for the catalogue.

    spectrum_exists : bool(optional)
        If the objects do not have spectral data, set this to False.
        In this case, load_data should only return photometry.

    photometry_exists : bool (optional)
        If the objects do not have photometric data for, set this to
        False. In this case, load_data should only return a
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

    def __init__(self, catalogue_IDs, fit_instructions, load_data,
                 filt_list=None, spectrum_exists=True,
                 photometry_exists=True, run=".", catalogue_redshifts=None,
                 fix_redshifts=False, save_phot=False, make_plots=False):

        self.run = run
        self.IDs = catalogue_IDs
        self.save_phot = save_phot
        self.make_plots = make_plots
        self.fix_redshifts = fix_redshifts
        self.load_data = load_data
        self.redshifts = catalogue_redshifts
        self.spectrum_exists = spectrum_exists
        self.fit_instructions = fit_instructions
        self.photometry_exists = photometry_exists
        self.filt_list = filt_list

        self.n_objects = catalogue_IDs.shape[0]

        if not os.path.exists(utils.working_dir + "/pipes/cats/" + self.run):
            os.mkdir(utils.working_dir + "/pipes/cats/" + self.run)

        np.savetxt(utils.working_dir + "/pipes/cats/" + self.run + "/all_IDs",
                   self.IDs)

    def fit(self, verbose=False, n_live=400, sampler="dynesty"):
        """ Run through the catalogue, only fitting objects which have
        not already been started by another thread. """
        cat_path = utils.working_dir + "/pipes/cats/"

        if os.path.exists(cat_path + self.run + "/kill"):
            call(["rm", cat_path + self.run + "/kill"])

        n = 0
        time0 = time.time()

        while True:

            for i in range(self.n_objects):

                if self.fix_redshifts:
                    self.fit_instructions["redshift"] = self.redshifts[i]

                elif isinstance(self.fix_redshifts, float):
                    z_range = (self.redshifts[i]-self.fix_redshifts/2.,
                               self.redshifts[i]+self.fix_redshifts/2.)

                    fit_instructions["redshift"] = z_range

                if not os.path.exists(utils.working_dir + "/pipes/cats/"
                                      + self.run+ "/" + str(int(self.IDs[i]))
                                      + ".lock"):

                    if os.path.exists(utils.working_dir + "/pipes/cats/"
                                      + self.run + "/kill"):

                        sys.exit("Kill command received")

                    np.savetxt(utils.working_dir + "/pipes/cats/" + self.run
                               + "/"+ str(int(self.IDs[i])) + ".lock",
                               np.array([0.]))


                    current_galaxy = galaxy(str(int(self.IDs[i])),
                                 self.load_data, filt_list=self.filt_list,
                                 spectrum_exists=self.spectrum_exists,
                                 photometry_exists=self.photometry_exists)

                    current_fit = fit(current_galaxy, self.fit_instructions,
                                      run=self.run)

                    """ Fit the object and make plots of the fit. """
                    current_fit.fit(verbose=verbose, n_live=n_live,
                                    sampler=sampler)

                    if self.make_plots:
                        current_fit.plot_fit()
                        current_fit.plot_corner()
                        current_fit.plot_1d_posterior()

                        if "polynomial" in self.fit_instructions.keys():
                            current_fit.plot_poly()

                    if n == 10:
                        n = 0
                        time0 = time.time()

                    """ Take care of output catalogue(s) """
                    if n == 0:
                        len_outcat = 3*current_fit.ndim + 20
                        outcat = np.zeros((10, len_outcat))
                        np.savetxt(utils.working_dir + "/pipes/cats/"
                                   + self.run + "/" + self.run + ".txt"
                                   + str(time0), outcat)

                        if self.save_phot:
                            bands = current_galaxy.photometry.shape[0]
                            photcat = np.zeros((10, 1 + 2*bands))
                            np.savetxt(utils.working_dir + "/pipes/cats/"
                                       + self.run + "/" + self.run 
                                       + "_phot.txt"+ str(time0), photcat)

                    outcat[n, 0] = current_fit.galaxy.ID

                    post = current_fit.posterior

                    outcat[n, -19] = np.percentile(post["UVJ"][:, 0]
                                                      - post["UVJ"][:, 1], 16)
                    outcat[n, -18] = np.percentile(post["UVJ"][:, 0]
                                                      - post["UVJ"][:, 1], 50)
                    outcat[n, -17] = np.percentile(post["UVJ"][:, 0]
                                                      - post["UVJ"][:, 1], 84)

                    outcat[n, -16] = np.percentile(post["UVJ"][:, 1]
                                                      - post["UVJ"][:, 2], 16)

                    outcat[n, -15] = np.percentile(post["UVJ"][:, 1]
                                                      - post["UVJ"][:, 2], 50)

                    outcat[n, -14] = np.percentile(post["UVJ"][:, 1]
                                                      - post["UVJ"][:, 2], 84)

                    outcat[n, -13] = np.percentile(post["tmw"], 16)
                    outcat[n, -12] = np.percentile(post["tmw"], 50)
                    outcat[n, -11] = np.percentile(post["tmw"], 84)

                    outcat[n, -10] = np.percentile(post["sfr"], 16)
                    outcat[n, -9] = np.percentile(post["sfr"], 50)
                    outcat[n, -8] = np.percentile(post["sfr"], 84)

                    living_mass = post["mass"]["total"]["living"]
                    outcat[n, -7] = np.percentile(living_mass, 16)
                    outcat[n, -6] = np.percentile(living_mass, 50)
                    outcat[n, -5] = np.percentile(living_mass, 84)

                    if self.redshifts is not None:
                        outcat[n, -4] = self.redshifts[i]

                    outcat[n, -3] = post["log_evidence"]
                    outcat[n, -2] = post["log_evidence_err"]
                    outcat[n, -1] = post["min_chisq_reduced"]

                    for j in range(current_fit.ndim):
                        conf_int = post["confidence_interval"]
                        param = current_fit.fit_params[j]
                        outcat[n, 1 + 3*j] = conf_int[param][0]
                        outcat[n, 2 + 3*j] = post["median"][param]
                        outcat[n, 3 + 3*j] = conf_int[param][1]

                    if self.save_phot:
                        photcat[n, 0] = current_fit.galaxy.ID
                        photcat[n, 1:1+bands] = current_galaxy.photometry[:, 1]
                        median_phot = np.percentile(post["photometry"],
                                                    50, axis=1)
                        photcat[n, 1+bands:] = median_phot

                    """ Set up the header for the output catalogue. """
                    fileheader = "ID "

                    for j in range(current_fit.ndim):
                        fileheader += (current_fit.fit_params[j] + "_16 "
                                       + current_fit.fit_params[j] + "_median "
                                       + current_fit.fit_params[j] + "_84 ")

                    fileheader += ("UVcolour_16 UVcolour_median UVcolour_84"
                                   + " VJcolour_16 VJcolour_median VJcolour_84"
                                   + " tmw_16 tmw_median tmw_84 sfr_16"
                                   + " sfr_median sfr_84 mstar_liv_16"
                                   + " mstar_liv_median mstar_liv_84 z_input"
                                   + " log_evidence"
                                   + " log_evidence_err"
                                   + " min_chisq_reduced")

                    """ Check to see if the kill switch has been set,
                    and if so stop here. """
                    if os.path.exists(utils.working_dir + "/pipes/cats/"
                                      + self.run + "/kill"):

                        sys.exit("Kill command received")

                    """ If not, save the updated output catalogue. """
                    np.savetxt(utils.working_dir + "/pipes/cats/" + self.run
                               + "/"+ self.run + ".txt" + str(time0), outcat,
                               header=fileheader)

                    if self.save_phot:
                        np.savetxt(utils.working_dir + "/pipes/cats/"
                                   + self.run + "/" + self.run + "_phot.txt"
                                   + str(time0),photcat)

                    n += 1

                    merge_cat(self.run)

            else:

                break


def merge_cat(run, mode="merge"):
    """ Compile all the sub-catalogues into one output catalogue,
    optionally stop all running processes and delete incomplete object
    posteriors. """

    if mode == "clean":
        call(["touch", utils.working_dir + "/pipes/cats/" + run + "/kill"])

    outcats = {}
    outphotcats = {}

    # Generate lists of files to merge
    files = glob.glob(utils.working_dir + "/pipes/cats/"
                      + run + "/" + run + ".txt*")

    photfiles = glob.glob(utils.working_dir + "/pipes/cats/" + run + "/"
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

    all_IDs = np.loadtxt(utils.working_dir + "/pipes/cats/" + run + "/all_IDs")

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
        np.savetxt(utils.working_dir + "/pipes/cats/" + run + "/" + run
                   + ".txt_clean", outcat_final[mask], header=header[2:-1])

        if merge_phot:
            mask = ((outphotcat_final[:, 1] != 0.)
                    & (outphotcat_final[:, 2] != 0.))

            np.savetxt(utils.working_dir + "/pipes/cats/" + run + "/" + run
                       + "_phot.txt_clean", outphotcat_final[mask])

        os.chdir("pipes/cats/" + run)
        lock_files = glob.glob("*.lock")
        os.chdir(utils.working_dir)

        IDs = np.loadtxt(utils.working_dir + "/pipes/cats/" + run + "/" + run
                         + ".txt_clean", usecols=(0, 1, 2))

        IDs = IDs[(IDs[:, 1] != 0.) & (IDs[:, 2] != 0.), 0]

        for lock_file in lock_files:
            if float(lock_file[:-5]) not in IDs:

                call(["rm", utils.working_dir + "/pipes/cats/" + run + "/"
                      + lock_file])

                files_toremove = glob.glob(utils.working_dir
                                            + "/pipes/posterior/"+ run + "/"
                                            + lock_file[:-5] + "*")

                for toremove in files_toremove:
                    call(["rm", toremove])

    mask = (outcat_final[:, 1] != 0.) & (outcat_final[:, 2] != 0.)
    np.savetxt(utils.working_dir + "/pipes/cats/" + run + ".cat",
               outcat_final[mask], header=header[2: -1])

    if merge_phot:
        mask = ((outphotcat_final[:, 1] != 0.)
                & (outphotcat_final[:, 2] != 0.))

        np.savetxt(utils.working_dir + "/pipes/cats/" + run + "_phot.cat",
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

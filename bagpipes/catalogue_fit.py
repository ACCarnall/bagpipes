from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os
import time
import glob
import pandas as pd

from astropy.io import fits
from subprocess import call

from . import utils
from . import plotting
from .galaxy import galaxy
from .fit import fit


class catalogue_fit:

    """ Fit a model to a catalogue of galaxies.

    Parameters
    ----------

    cat_IDs : list
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

    spectrum_exists : bool(optional)
        If the objects do not have spectral data, set this to False.
        In this case, load_data should only return photometry.

    photometry_exists : bool (optional)
        If the objects do not have photometric data for, set this to
        False. In this case, load_data should only return a spectrum.

    run : string (optional)
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    cat_filt_list : list (optional)
        The filt_list, or list of filt_lists for the catalogue.

    vary_filt_list : bool (optional)
        If True, changes the filter list for each object. When True,
        each entry in cat_filt_list is expected to be a different filter
        list corresponding to each object in the catalogue.

    cat_redshifts : list (optional)
        A list of redshift values associated with the objects to be
        fitted. Same length as Catalogue_IDs.

    fix_redshifts : bool or float (optional)
        If False (default), whatever instructions given in the input
        fit_instructions for redshift will be applied. if True,
        redshifts are fixed to the values specified in  cat_redshifts.
        If a float the redshift will be varied  within this range either
        side of the value specified in cat_redshifts.

    make_plots : bool (optional)
        If True, spectral and corner plots will be made for each object.

    """

    def __init__(self, cat_IDs, fit_instructions, load_data,
                 spectrum_exists=True, photometry_exists=True, run=".",
                 cat_filt_list=None, vary_filt_list=False, cat_redshifts=None,
                 fix_redshifts=False, make_plots=False):

        self.IDs = np.array(cat_IDs).astype(str)
        self.fit_instructions = fit_instructions
        self.load_data = load_data

        self.spectrum_exists = spectrum_exists
        self.photometry_exists = photometry_exists
        self.run = run
        self.cat_filt_list = cat_filt_list
        self.vary_filt_list = vary_filt_list
        self.redshifts = cat_redshifts
        self.fix_redshifts = fix_redshifts

        self.make_plots = make_plots

        self.n_objects = len(self.IDs)

        utils.make_dirs()

        if not os.path.exists(utils.working_dir + "/pipes/cats/" + self.run):
            os.mkdir(utils.working_dir + "/pipes/cats/" + self.run)

        np.savetxt(utils.working_dir + "/pipes/cats/" + self.run + "/all_IDs",
                   self.IDs, fmt="%s")

    def fit(self, verbose=False, n_live=400, sampler="dynesty"):
        """ Run through the catalogue, only fitting objects which have
        not already been started by another thread. """
        cat_path = utils.working_dir + "/pipes/cats/"

        if os.path.exists(cat_path + self.run + "/kill"):
            call(["rm", cat_path + self.run + "/kill"])

        n = 0
        time0 = time.time()

        for i in range(self.n_objects):

            if isinstance(self.fix_redshifts, float):
                z_range = (self.redshifts[i]-self.fix_redshifts,
                           self.redshifts[i]+self.fix_redshifts)

                self.fit_instructions["redshift"] = z_range

            elif self.fix_redshifts:
                self.fit_instructions["redshift"] = self.redshifts[i]

            if not os.path.exists(utils.working_dir + "/pipes/cats/"
                                  + self.run + "/" + str(self.IDs[i])
                                  + ".lock"):

                if os.path.exists(utils.working_dir + "/pipes/cats/"
                                  + self.run + "/kill"):

                    sys.exit("Kill command received")

                np.savetxt(utils.working_dir + "/pipes/cats/" + self.run
                           + "/" + str(self.IDs[i]) + ".lock", np.array([0.]))

                if self.vary_filt_list:
                    self.filt_list = self.cat_filt_list[i]

                else:
                    self.filt_list = self.cat_filt_list

                c_galaxy = galaxy(str(self.IDs[i]),
                                  self.load_data,
                                  filt_list=self.filt_list,
                                  spectrum_exists=self.spectrum_exists,
                                  photometry_exists=self.photometry_exists)

                current_fit = fit(c_galaxy, self.fit_instructions,
                                  run=self.run)

                """ Fit the object and make plots of the fit. """
                current_fit.fit(verbose=verbose, n_live=n_live,
                                sampler=sampler)

                if self.make_plots:
                    current_fit.plot_fit()
                    current_fit.plot_corner()
                    current_fit.plot_1d_posterior()
                    current_fit.plot_sfh()

                    if "polynomial" in self.fit_instructions.keys():
                        current_fit.plot_poly()

                if n == 10:
                    n = 0
                    time0 = time.time()

                """ Take care of output catalogue(s) """
                if n == 0:

                    out_cat_names = ["#ID"]
                    extra_vars = ["UVcolour", "VJcolour", "tmw", "mwa",
                                  "sfr", "stellar_mass"]

                    variables = current_fit.fit_params + extra_vars

                    for var in variables:
                        out_cat_names += [var + "_16",
                                          var + "_50",
                                          var + "_84"]

                    out_cat_names += ["input_redshift", "log_evidence",
                                      "log_evidence_err", "min_chisq_reduced"]

                    outcat = pd.DataFrame(np.zeros((10, len(out_cat_names))),
                                          columns=out_cat_names)

                    outcat.to_csv(utils.working_dir + "/pipes/cats/"
                                  + self.run + "/" + self.run + ".txt"
                                  + str(time0), sep="\t", index=False)

                outcat["#ID"][n] = current_fit.galaxy.ID

                post = current_fit.posterior

                for j in range(len(variables)):
                    var = variables[j]
                    if var == "stellar_mass":
                        quant = post["mass"]["total"]["living"]

                    elif var == "UVcolour":
                        quant = post["UVJ"][:, 0] - post["UVJ"][:, 1]

                    elif var == "VJcolour":
                        quant = post["UVJ"][:, 1] - post["UVJ"][:, 2]

                    else:
                        quant = post[var]

                    percentiles = [16, 50, 84]
                    for p in percentiles:
                        colname = var + "_" + str(p)
                        outcat.loc[n, colname] = np.percentile(quant, p)

                if self.redshifts is not None:
                    outcat.loc[n, "input_redshift"] = self.redshifts[i]

                outcat.loc[n, "log_evidence"] = post["log_evidence"]
                outcat.loc[n, "log_evidence_err"] = post["log_evidence_err"]
                outcat.loc[n, "min_chisq_reduced"] = post["min_chisq_reduced"]

                """ Check to see if the kill switch has been set. """
                if os.path.exists(utils.working_dir + "/pipes/cats/"
                                  + self.run + "/kill"):

                    sys.exit("Kill command received")

                """ Save the updated output catalogue. """
                outcat.to_csv(utils.working_dir + "/pipes/cats/" + self.run
                              + "/" + self.run + ".txt" + str(time0), sep="\t",
                              index=False)

                n += 1

                merge_cat(self.run)


def merge_cat(run, mode="merge"):
    """ Compile all the sub-catalogues into one output catalogue,
    optionally stop all running processes and delete incomplete object
    posteriors with the "clean" mode. """

    if mode == "clean":
        call(["touch", utils.working_dir + "/pipes/cats/" + run + "/kill"])

    outcats = []
    outphotcats = []

    # Generate lists of files to merge
    files = glob.glob(utils.working_dir + "/pipes/cats/"
                      + run + "/" + run + ".txt*")

    photfiles = glob.glob(utils.working_dir + "/pipes/cats/" + run + "/"
                          + run + "_phot.txt*")

    header = " ".join((open(files[0]).readline()[:-1]).split("\t"))

    # Load up files
    for file in files:
        while True:
            try:
                outcats.append(pd.read_table(file, delimiter="\t",
                                             names=header.split(), skiprows=1))

                outcats[-1].index = outcats[-1]["#ID"]
                break

            except ValueError:
                time.sleep(1)

    # Generate files to merge outputs into
    all_IDs = np.loadtxt(utils.working_dir + "/pipes/cats/"
                         + run + "/all_IDs", dtype=str)

    finalcat = pd.DataFrame(np.zeros((all_IDs.shape[0], outcats[0].shape[1])),
                            columns=header.split(), index=all_IDs)

    finalcat.loc[:, "#ID"] = all_IDs

    # Merge outputs into final catalogue
    for ind in finalcat.index:
        for outcat in outcats:
            if ind in outcat.index:
                finalcat.loc[ind, :] = outcat.loc[ind, :]
                break

        else:
            finalcat.loc[ind, "#ID"] = np.nan

    finalcat = finalcat.groupby(finalcat["#ID"].isnull()).get_group(False)
    finalcat.to_csv(utils.working_dir + "/pipes/cats/" + run + ".cat",
                    sep="\t", index=False)

    # If mode is clean, remove all input catalogues and replace with a
    # merged one, also delete objects which are in progress
    if mode == "clean":
        for file in files:
            call(["rm",  file])

        finalcat.to_csv(utils.working_dir + "/pipes/cats/" + run + "/" + run
                        + ".txt_clean", sep="\t", index=False)

        os.chdir("pipes/cats/" + run)
        lock_files = glob.glob("*.lock")
        os.chdir(utils.working_dir)

        for lock_file in lock_files:
            if lock_file[:-5] not in finalcat.loc[:, "#ID"]:
                files_toremove = [utils.working_dir + "/pipes/cats/"
                                  + run + "/" + lock_file]

                files_toremove += glob.glob(utils.working_dir
                                            + "/pipes/posterior/" + run + "/"
                                            + lock_file[:-5] + "*")

                for file in files_toremove:
                    call(["rm", file])

    print("Bagpipes:", finalcat.shape[0], "out of",
          all_IDs.shape[0], "objects completed.")

    if mode == "clean":
        print("Bagpipes: Partially completed objects reset.")


def clean_cat(run):
    """ Run compile_cat with the clean option enabled to kill running
    processes and delete progress for uncompleted objects. """
    merge_cat(run, mode="clean")

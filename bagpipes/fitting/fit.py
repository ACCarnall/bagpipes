from __future__ import print_function, division, absolute_import

import numpy as np
import os
import time
import warnings
import h5py

from copy import deepcopy

try:
    import pymultinest as pmn

except (ImportError, RuntimeError, SystemExit) as e:
    print("Bagpipes: PyMultiNest import failed, fitting will be unavailable.")

try:
    from nautilus import Sampler

except (ImportError, RuntimeError, SystemExit) as e:
    pass

# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

    if size == 1:
        pool = None

    else:
        pool = MPIPoolExecutor(size)

except ImportError:
    rank = 0
    pool = None

from .. import utils
from .. import plotting

from .fitted_model import fitted_model
from .posterior import posterior


class fit(object):
    """ Top-level class for fitting models to observational data.
    Interfaces with MultiNest to sample from the posterior distribution
    of a fitted_model object. Performs loading and saving of results.

    Parameters
    ----------

    galaxy : bagpipes.galaxy
        A galaxy object containing the photomeric and/or spectroscopic
        data you wish to fit.

    fit_instructions : dict
        A dictionary containing instructions on the kind of model which
        should be fitted to the data.

    run : string - optional
        The subfolder into which outputs will be saved, useful e.g. for
        fitting more than one model configuration to the same data.

    time_calls : bool - optional
        Whether to print information on the average time taken for
        likelihood calls.

    n_posterior : int - optional
        How many equally weighted samples should be generated from the
        posterior once fitting is complete. Default is 500.
    """

    def __init__(self, galaxy, fit_instructions, run=".", time_calls=False,
                 n_posterior=500):

        self.run = run
        self.galaxy = galaxy
        self.fit_instructions = deepcopy(fit_instructions)
        self.n_posterior = n_posterior

        # Set up the directory structure for saving outputs.
        if rank == 0:
            utils.make_dirs(run=run)

        # The base name for output files.
        self.fname = "pipes/posterior/" + run + "/" + self.galaxy.ID + "_"

        # A dictionary containing properties of the model to be saved.
        self.results = {}

        # If a posterior file already exists load it.
        if os.path.exists(self.fname[:-1] + ".h5"):
            file = h5py.File(self.fname[:-1] + ".h5", "r")

            self.posterior = posterior(self.galaxy, run=run,
                                       n_samples=n_posterior)

            fit_info_str = file.attrs["fit_instructions"]
            fit_info_str = fit_info_str.replace("array", "np.array")
            fit_info_str = fit_info_str.replace("float", "np.float")
            self.fit_instructions = eval(fit_info_str)

            for k in file.keys():
                self.results[k] = np.array(file[k])
                if np.sum(self.results[k].shape) == 1:
                    self.results[k] = self.results[k][0]

            if rank == 0:
                print("\nResults loaded from " + self.fname[:-1] + ".h5\n")

        # Set up the model which is to be fitted to the data.
        self.fitted_model = fitted_model(galaxy, self.fit_instructions,
                                         time_calls=time_calls)

    def fit(self, verbose=False, n_live=400, use_MPI=True, sampler="multinest"):
        """ Fit the specified model to the input galaxy data.

        Parameters
        ----------

        verbose : bool - optional
            Set to True to get progress updates from the sampler.

        n_live : int - optional
            Number of live points: reducing speeds up the code but may
            lead to unreliable results.
        """

        if "lnz" in list(self.results):
            if rank == 0:
                print("Fitting not performed as results have already been"
                      + " loaded from " + self.fname[:-1] + ".h5. To start"
                      + " over delete this file or change run.\n")

            return

        if rank == 0 or not use_MPI:
            print("\nBagpipes: fitting object " + self.galaxy.ID + "\n")

            start_time = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if sampler == "multinest":
                pmn.run(self.fitted_model.lnlike,
                        self.fitted_model.prior.transform,
                        self.fitted_model.ndim, n_live_points=n_live,
                        importance_nested_sampling=False, verbose=verbose,
                        sampling_efficiency="model",
                        outputfiles_basename=self.fname, use_MPI=use_MPI)

            elif sampler == "nautilus":
                n_sampler = Sampler(self.fitted_model.prior.transform,
                                        self.fitted_model.lnlike, n_live=n_live,
                                        n_networks=1, pool=pool,
                                        n_dim=self.fitted_model.ndim,
                                        filepath=self.fname + "temp.h5")

                n_sampler.run(verbose=verbose)

        if rank == 0 or not use_MPI:
            runtime = time.time() - start_time

            print("\nCompleted in " + str("%.1f" % runtime) + " seconds.\n")

            # Load MultiNest outputs and save basic quantities to file.
            if sampler == "multinest":
                samples2d = np.loadtxt(self.fname + "post_equal_weights.dat")
                lnz_line = open(self.fname + "stats.dat").readline().split()
                self.results["samples2d"] = samples2d[:, :-1]
                self.results["lnlike"] = samples2d[:, -1]
                self.results["lnz"] = float(lnz_line[-3])
                self.results["lnz_err"] = float(lnz_line[-1])

            elif sampler == "nautilus":
                samples2d, log_w, log_l = n_sampler.posterior(equal_weight=True)
                self.results["samples2d"] = samples2d
                self.results["lnlike"] = log_l
                self.results["lnz"] = n_sampler.evidence()
                self.results["lnz_err"] = -99

            self.results["median"] = np.median(samples2d, axis=0)
            self.results["conf_int"] = np.percentile(self.results["samples2d"],
                                                     (16, 84), axis=0)

            file = h5py.File(self.fname[:-1] + ".h5", "w")

            # This is necessary for converting large arrays to strings
            np.set_printoptions(threshold=10**7)
            file.attrs["fit_instructions"] = str(self.fit_instructions)
            np.set_printoptions(threshold=10**4)

            for k in self.results.keys():
                file.create_dataset(k, data=self.results[k])

            self.results["fit_instructions"] = self.fit_instructions

            file.close()

            os.system("rm " + self.fname + "*")

            self._print_results()

            # Create a posterior object to hold the results of the fit.
            self.posterior = posterior(self.galaxy, run=self.run,
                                       n_samples=self.n_posterior)

    def _print_results(self):
        """ Print the 16th, 50th, 84th percentiles of the posterior. """

        print("{:<25}".format("Parameter")
              + "{:>31}".format("Posterior percentiles"))

        print("{:<25}".format(""),
              "{:>10}".format("16th"),
              "{:>10}".format("50th"),
              "{:>10}".format("84th"))

        print("-"*58)

        for i in range(self.fitted_model.ndim):
            print("{:<25}".format(self.fitted_model.params[i]),
                  "{:>10.3f}".format(self.results["conf_int"][0, i]),
                  "{:>10.3f}".format(self.results["median"][i]),
                  "{:>10.3f}".format(self.results["conf_int"][1, i]))

        print("\n")

    def plot_corner(self, show=False, save=True):
        return plotting.plot_corner(self, show=show, save=save)

    def plot_1d_posterior(self, show=False, save=True):
        return plotting.plot_1d_posterior(self, show=show, save=save)

    def plot_sfh_posterior(self, show=False, save=True, colorscheme="bw"):
        return plotting.plot_sfh_posterior(self, show=show, save=save,
                                           colorscheme=colorscheme)

    def plot_spectrum_posterior(self, show=False, save=True):
        return plotting.plot_spectrum_posterior(self, show=show, save=save)

    def plot_calibration(self, show=False, save=True):
        return plotting.plot_calibration(self, show=show, save=save)

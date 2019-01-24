from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os

from copy import deepcopy

from ..models.model_galaxy import model_galaxy
from ..input.galaxy import galaxy

from .. import utils

class mock(object):

    def __init__(self, model_components, etc_parameters,
                 input_spec=None, input_phot=None):

        if "MOONS_ETC_PATH" in list(os.environ):
            self.etc_path = os.environ["MOONS_ETC_PATH"]
            if self.etc_path.endswith("/moons_etc"):
                self.etc_path = os.environ["MOONS_ETC_PATH"][:-10]

        else:
            raise EnvironmentError("Set MOONS_ETC_PATH environment variable.")

        self.etc_parameters = self._setup_etc_params(deepcopy(etc_parameters))

        bands = ["i_sdss", "J_WFC3", "H_WFC3"]
        b = np.argmax(np.isin(["RI", "YJ", "H"],
                      self.etc_parameters["channel"]))

        self.wavs = np.loadtxt(utils.install_dir + "/moons/wavs/"
                          + self.etc_parameters["channel"] + ".txt")*10**4

        if input_spec is None:
            self.model = model_galaxy(model_components, spec_wavs=self.wavs)
            self.spectrum = self.model.spectrum

            self.model_phot = model_galaxy(model_components, phot_units="mujy",
                                           filt_list=["moons/filters/sdss_i",
                                                      "moons/filters/f125w",
                                                      "moons/filters/f160w"])

            mags = 23.9 - 2.5*np.log10(self.model_phot.photometry)
            mag = 23.9 - 2.5*np.log10(self.model_phot.photometry[b])

        else:
            self.spectrum = np.c_[self.wavs, input_spec]
            mags = 23.9 - 2.5*np.log10(input_phot)
            mag = mags[b]

        self.magnitudes = dict(zip(bands, mags))

        self.etc_parameters["AB"] = str(np.round(mag, 2))

        self._run_etc()

        self.observation = galaxy("moons_mock_obs", self._load_spec,
                                  photometry_exists=False)

    def _setup_etc_params(self, etc_parameters):
        self.params = ["resolution", "channel", "atm_corr", "AB", "extended",
                       "template", "emlineW", "emlineFWHM", "emlineF",
                       "redshift", "seeing", "airmass", "stray", "skyres",
                       "NDIT", "DIT"]

        default_vals = ["LR", "RI", "1.2", "20.0", "0", "etc_spec.txt",
                        "0.0", "0.0", "0.0", "0.0", "1.0", "1.4", "1.0", "0.0",
                        "1", "600"]

        if "redshift" in list(etc_parameters):
            etc_parameters["redshift"] = 0.
            print("Bagpipes already redshifts the input spectrum, please enter"
                  + " your desired redshift in model_components rather than"
                  + " etc_parameters. The etc_parameters redshift will be set"
                  + " automatically to zero.")

        if "AB" in list(etc_parameters):
            print("Bagpipes will automatically calculate the relevant "
                  + "magnitude for the galaxy with parameters specified in "
                  + "model_components. This can be scaled by changing the "
                  + "stellar mass. The calculated magnitude can be accessed "
                  + "under mock.etc_parameters['AB'].")

        for i in range(len(self.params)):
            param = self.params[i]
            if param not in list(etc_parameters):
                etc_parameters[param] = default_vals[i]

        return etc_parameters

    def _run_etc(self):
        command = "./moons_etc batch "

        for param in self.params:
            command += str(self.etc_parameters[param]) + " "

        print("Bagpipes: Calling MOONS ETC:", command, "\n")

        os.chdir(self.etc_path)
        np.savetxt("etc_spec.txt", self.spectrum)
        os.system(command)
        self.snr = np.loadtxt("Sensitivity_table.txt")
        os.chdir(utils.working_dir)

        print("Bagpipes: MOONS ETC finished")

    def _load_spec(self, ID):
        spec = np.c_[self.spectrum,
                     self.spectrum[:, 1]/self.snr[:, 1]]

        spec[:, 1] += np.random.randn(spec.shape[0])*spec[:, 2]

        return spec

from __future__ import print_function, division, absolute_import

import numpy as np
import sys

from numpy.polynomial.chebyshev import chebval as cheb

from . import utils
from .star_formation_history import *
from . import plotting

# Ignore division by zero and overflow warnings
np.seterr(divide='ignore', invalid='ignore', over="ignore")


class model_galaxy:
    """ Build a model galaxy spectrum. Note, at least one of filt_list
    or spec_wavs must be defined.

    Parameters
    ----------

    model_components : dict
        A dictionary containing information about the model you wish to
        generate.

    filt_list : list - optional
        A list of paths to filter curve files, which should contain a 
        column of wavelengths in angstroms followed by a column of 
        transmitted fraction values. Only required if photometric output
        is desired.

    spec_wavs : array - optional
        An array of wavelengths at which spectral fluxes should be
        returned. Only required of spectroscopic output is desired.

    spec_units : str - optional
        The units the output spectrum will be returned in. Default is
        "ergscma" for ergs per second per centimetre squared per
        angstrom, can also be set to "mujy" for microjanskys.

    phot_units : str - optional
        The units the output spectrum will be returned in. Default is
        "ergscma" for ergs per second per centimetre squared per
        angstrom, can also be set to "mujy" for microjanskys.
    """

    def __init__(self, model_components, filt_list=None, spec_wavs=None,
                 spec_units="ergscma", phot_units="ergscma"):

        if utils.model_type not in list(utils.ages):
            utils.set_model_type(utils.model_type)

        if spec_wavs is None and filt_list is None:
            sys.exit("Bagpipes: Specify either filt_list or spec_wavs.")

        self.model_comp = model_components
        self.filt_list = filt_list
        self.spec_wavs = spec_wavs
        self.spec_units = spec_units
        self.phot_units = phot_units

        self.dust_on = ("dust" in list(self.model_comp))
        self.nebular_on = ("nebular" in list(self.model_comp))
        self.extra_dust_on = (self.dust_on
                              and "eta" in list(self.model_comp["dust"]))

        # modelgrids: stores stellar grids resampled onto chosen_wavs
        self.stellar_grids = {}

        # cloudygrids: stores nebular grids resampled onto chosen_wavs
        self.nebular_grids = {}

        # UVJ_filt_names: UVJ filter names, see get_restframe_UVJ.
        self.UVJ_filt_names = None

        # Loads filter curves from filt_list and finds max and min wavs.
        if self.filt_list is not None:
            self._load_filter_curves()

        # Calculate chosen_wavs, a common wavelength sampling.
        self._get_chosen_wavs()

        # Resample filter curves to chosen_wavs.
        if self.filt_list is not None:
            self._resample_filter_curves()

        # Resample the igm grids onto chosen_wavs.
        self._resample_igm_grids()

        # Set up dust model for emission lines.
        if self.dust_on:
            self.k_line = self._get_dust_model(utils.line_wavs)
            self.k_cont = self._get_dust_model(self.chosen_wavs)

        # keep_ionizing: stops the code removing all flux below 912A.
        if ("keep_ionizing" in list(self.model_comp)
                and self.model_comp["keep_ionizing"]):

            self.keep_ionizing = True

        else:
            self.keep_ionizing = False

        # sfh_components: List of the SFH components for this model
        self.sfh_components = []
        for comp in list(self.model_comp):
            if (not comp.startswith(("dust", "nebular", "polynomial"))
                    and isinstance(self.model_comp[comp], dict)):

                self.sfh_components.append(comp)

        # Check the model has at least one SFH component.
        if len(self.sfh_components) == 0:
            sys.exit("Bagpipes: No SFH components passed to Model_Galaxy.")

        self.update(self.model_comp)

    def _load_filter_curves(self):
        """ Loads filter files for the specified filt_list, truncates
        zeros from their edges and calculates effective wavelength
        values which are added to eff_wavs. """

        # self.filt_dict: a dict containing the raw filter files
        self.filt_dict = {}
        for filt in self.filt_list:
            self.filt_dict[filt] = np.loadtxt(filt, usecols=(0, 1))

            # Get rid of trailing zeros at ends of the filter files
            while self.filt_dict[filt][0, 1] == 0.:
                self.filt_dict[filt] = self.filt_dict[filt][1:, :]

            while self.filt_dict[filt][-1, 1] == 0.:
                self.filt_dict[filt] = self.filt_dict[filt][:-1, :]

        # min_phot_wav, max_phot_wav: the min and max wavelengths
        # covered by any of the photometric bands.
        self.min_phot_wav = 9.9*10**99
        self.max_phot_wav = 0.

        for filt in self.filt_list:
            min_wav = (self.filt_dict[filt][0, 0]
                       - 2*(self.filt_dict[filt][1, 0]
                       - self.filt_dict[filt][0, 0]))

            max_wav = (self.filt_dict[filt][-1, 0]
                       + 2*(self.filt_dict[filt][-1, 0]
                       - self.filt_dict[filt][-2, 0]))

            if min_wav < self.min_phot_wav:
                self.min_phot_wav = min_wav

            if max_wav > self.max_phot_wav:
                self.max_phot_wav = max_wav

        # eff_wavs: effective wavelengths of each filter curve
        self.eff_wavs = np.zeros(len(self.filt_list))

        for i in range(len(self.filt_list)):
            filt = self.filt_list[i]
            dlambda = utils.make_bins(self.filt_dict[filt][:, 0])[1]
            filt_weights = dlambda*self.filt_dict[filt][:, 1]
            self.eff_wavs[i] = np.sqrt(np.sum(filt_weights)
                                       / np.sum(filt_weights
                                       / self.filt_dict[filt][:, 0]**2))

    def _resample_filter_curves(self):
        """ Resamples the filter curves onto the chosen_wavs and creates
         a 2D grid filt_array of all filter curves. """

        # filt_rest: An array to contain the rest frame sampled filter
        # profiles used to generate the model photometry.
        self.filt_rest = np.zeros((self.chosen_wavs.shape[0],
                                   len(self.filt_list)))

        for i in range(len(self.filt_list)):
            filt = self.filt_list[i]
            self.filt_rest[:, i] = np.interp(self.chosen_wavs,
                                             self.filt_dict[filt][:, 0],
                                             self.filt_dict[filt][:, 1],
                                             left=0, right=0)

        # filt_array: An array to contain the resampled filter profiles
        # used to generate the model photometry.
        self.filt_array = np.zeros((self.chosen_wavs.shape[0],
                                    len(self.filt_list)))

        # model_widths: the widths of the bins with midp chosen_wavs.
        model_widths = utils.make_bins(self.chosen_wavs)[1]

        # wav_widths: An array containing the relative widths*wavs
        # for each point in the spectrum.
        self.wav_widths = model_widths*self.chosen_wavs

    def _get_chosen_wavs(self):
        """ Sets up chosen_wavs: sets up an optimal wavelength sampling
        for the models within the class to be interpolated onto to
        maximise the speed of the code. """

        x = [1.]

        if self.spec_wavs is None:
            self.max_wavs = [self.min_phot_wav/(1.+utils.max_redshift),
                             1.01*self.max_phot_wav, 10**8]

            self.R = [10., 100., 10.]

        elif self.filt_list is None:
            self.max_wavs = [self.spec_wavs[0]/(1.+utils.max_redshift),
                             self.spec_wavs[-1], 10**8]

            self.R = [10., 600., 10.]

        else:
            if (self.spec_wavs[0] > self.min_phot_wav
                    and self.spec_wavs[-1] < self.max_phot_wav):

                self.max_wavs = [self.min_phot_wav/(1.+utils.max_redshift),
                                 self.spec_wavs[0]/(1.+utils.max_redshift),
                                 self.spec_wavs[-1],
                                 self.max_phot_wav, 10**8]

                self.R = [10., 100., 600., 100., 10.]

            elif (self.spec_wavs[0] < self.min_phot_wav
                  and self.spec_wavs[-1] < self.max_phot_wav):

                self.max_wavs = [self.spec_wavs[0]/(1.+utils.max_redshift),
                                 self.spec_wavs[-1],
                                 self.max_phot_wav, 10**8]

                self.R = [10., 600., 100., 10.]

            if (self.spec_wavs[0] > self.min_phot_wav
                    and self.spec_wavs[-1] > self.max_phot_wav):

                self.max_wavs = [self.min_phot_wav/(1.+utils.max_redshift),
                                 self.spec_wavs[0]/(1.+utils.max_redshift),
                                 self.spec_wavs[-1], 10**8]

                self.R = [10., 100., 600., 10.]

        # Generate chosen_wavs from 1A to 10**8A.
        for i in range(len(self.R)):
            if i == len(self.R)-1 or self.R[i] > self.R[i+1]:
                while x[-1] < self.max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/self.R[i]))

            else:
                while x[-1]*(1.+0.5/self.R[i]) < self.max_wavs[i]:
                    x.append(x[-1]*(1.+0.5/self.R[i]))

        self.chosen_wavs = np.array(x)

    def _resample_igm_grids(self):
        """ Set up the igm grids on the same wavelength sampling as the
        nebular and stellar models. """

        self.igm_mask = (self.chosen_wavs > 911.8) & (self.chosen_wavs < 1220.)

        # set up igm_cont to contain transmission values for continuum
        # from 912A to 1216A as a function of redshift.
        self.igm_cont = np.zeros((utils.igm_redshifts.shape[0],
                                  self.chosen_wavs[self.igm_mask].shape[0]))

        # Resample from the raw grids held in the utils file.
        for i in range(utils.igm_redshifts.shape[0]):
            self.igm_cont[i, :] = np.interp(self.chosen_wavs[self.igm_mask],
                                            utils.igm_wavs,
                                            utils.igm_grid[i, :])

        # Set up igm_lines to contain a grid of transmission values for
        # emission lines from 912A to 1216A as a function of redshift.
        self.igm_lines = np.zeros((utils.igm_redshifts.shape[0],
                                   utils.line_wavs.shape[0]))

        for i in range(utils.igm_redshifts.shape[0]):
            self.igm_lines[i, :] = np.interp(utils.line_wavs, utils.igm_wavs,
                                             utils.igm_grid[i, :],
                                             left=0., right=1.)

    def _get_dust_model(self, wavs):
        """ Returns k values for chosen dust curve normalised to an Av
        of 1. """

        dust_type = self.model_comp["dust"]["type"]
        dust_path = utils.install_dir + "/pipes_models/dust/"

        if dust_type == "Calzetti":
            k_lam = np.loadtxt(dust_path + "Calzetti_2000.txt")

            return np.interp(wavs, k_lam[:, 0], k_lam[:, 1], right=0)

        elif dust_type == "Cardelli":
            k_lam = np.loadtxt(dust_path + "Cardelli_1989.txt")

            return np.interp(wavs, k_lam[:, 0], k_lam[:, 1], right=0)

        elif dust_type == "CF00":
            return wavs/5500.

        else:
            sys.exit("BAGPIPES: Dust type not recognised.")

    def _load_stellar_grid(self, zmet_ind):
        """ Loads stellar grid at specified metallicity, and resamples
        it to chosen_wavs. """

        grid = utils.load_stellar_grid(zmet_ind)

        # Resample model grid onto chosen_wavs.
        grid_res = np.zeros((grid.shape[0], self.chosen_wavs.shape[0]))

        for i in range(grid_res.shape[0]):
            grid_res[i, :] = np.interp(self.chosen_wavs,
                                       utils.gridwavs[utils.model_type],
                                       grid[i, :], left=0, right=0)

        # Add the resampled grid to the modelgrids dictionary
        self.stellar_grids[str(zmet_ind)] = grid_res

    def _load_cloudy_grid(self, zmet_val, logU):
        """ Loads Cloudy nebular continuum and emission lines at
        specified metallicity and logU. """

        raw_cont_grid, raw_line_grid = utils.load_cloudy_grid(zmet_val, logU)

        # Resample the nebular continuum onto the same wavelength basis
        # as the stellar grids have been resampled to.
        cont_grid = np.zeros((raw_cont_grid.shape[0],
                              self.chosen_wavs.shape[0]))

        for i in range(cont_grid.shape[0]):
            cont_grid[i, :] = np.interp(self.chosen_wavs,
                                        utils.gridwavs[utils.model_type],
                                        raw_cont_grid[i, :], left=0, right=0)

        # Add the nebular lines to the resampled nebular continuum
        for i in range(utils.line_wavs.shape[0]):
            ind = np.abs(self.chosen_wavs - utils.line_wavs[i]).argmin()
            if ind != 0 and ind != self.chosen_wavs.shape[0]-1:
                width = (self.chosen_wavs[ind+1] - self.chosen_wavs[ind-1])/2
                cont_grid[:, ind] += raw_line_grid[:, i]/width

        # Add the finished cloudy grid to the cloudygrids dictionary
        self.nebular_grids[str(zmet_val) + str(logU)] = cont_grid

    def _combine_stellar_grids(self, zmet_wts):
        """ Combines stellar grids in metallicity to create grids which
        are ready to be multiplied by the sfh. """

        zmet_vals = utils.zmet_vals[utils.model_type]

        # Check relevant stellar grids have been loaded.
        for zmet_val in zmet_vals[zmet_wts != 0.]:
            if str(zmet_val) not in list(self.stellar_grids):
                self._load_stellar_grid(zmet_val)

        # Create blank arrays for stellar grids.
        stellar_key = list(self.stellar_grids)[0]
        stellar_grid = np.zeros_like(self.stellar_grids[stellar_key])

        # Combine by metallicity to get age vs wavelength grids
        for zmet_val in zmet_vals[zmet_wts != 0.]:
            wt = zmet_wts[zmet_vals == zmet_val]
            stellar_grid += wt*self.stellar_grids[str(zmet_val)]

        return stellar_grid

    def _combine_neb_grids(self, zmet_wts):
        """ Combines nebular grids in metallicity and logU to create
        grids which are ready to be multiplied by the sfh. """
        zmet_vals = utils.zmet_vals[utils.model_type]

        # Figure out which nebular grids to use and their weights.
        logU = self.model_comp["nebular"]["logU"]

        if (logU < utils.logU_grid[0]) or (logU > utils.logU_grid[-1]):
            sys.exit("Bagpipes: logU outside of the nebular grid range")

        # Deal with logU exactly equal to smallest value in the grid
        elif logU == utils.logU_grid[0]:
            logU += 10**-10

        logU_vals = [np.min(utils.logU_grid[utils.logU_grid >= logU]),
                     np.max(utils.logU_grid[utils.logU_grid < logU])]

        logU_wts = [(logU - logU_vals[0])/(logU_vals[1] - logU_vals[0]), 0.]
        logU_wts[1] = 1. - logU_wts[1]

        # Check relevant nebular grids have been loaded.
        for logU_val in (logU_vals[1], logU_vals[0]):
            for zmet_val in zmet_vals[zmet_wts != 0.]:
                key = str(zmet_val) + str(logU_val)
                if key not in list(self.nebular_grids):
                    self._load_cloudy_grid(zmet_val, logU_val)

        # Create blank arrays for nebular grids.
        nebular_key = list(self.nebular_grids)[0]
        line_key = list(utils.allcloudylinegrids)[0]
        nebular_grid = np.zeros_like(self.nebular_grids[nebular_key])
        line_grid = np.zeros_like(utils.allcloudylinegrids[line_key])

        # Combine the relevant grids.
        for zmet_val in zmet_vals[zmet_wts != 0.]:
            wt = zmet_wts[zmet_vals == zmet_val]
            key_low = str(zmet_val) + str(logU_vals[0])
            key_high = str(zmet_val) + str(logU_vals[1])

            nebular_grid += wt*(logU_wts[1]*self.nebular_grids[key_high]
                                + logU_wts[0]*self.nebular_grids[key_low])

            line_grid += wt*(logU_wts[1]*utils.allcloudylinegrids[key_high]
                             + logU_wts[0]*utils.allcloudylinegrids[key_low])

        return nebular_grid, line_grid

    def _calculate_photometry(self):
        """ Resamples filter curves onto observed frame wavelengths and
        integrates over them to calculate photometric fluxes. """
        redshifted_wavs = self.chosen_wavs*(1.+self.model_comp["redshift"])

        for i in range(len(self.filt_list)):
            self.filt_array[:, i] = np.interp(redshifted_wavs,
                                              self.chosen_wavs,
                                              self.filt_rest[:, i],
                                              left=0, right=0)

        spec_energy = np.expand_dims(self.spectrum_full*self.wav_widths,
                                     axis=1)

        filt_weights = self.filt_array*np.expand_dims(self.wav_widths, axis=1)

        return np.squeeze(np.sum(spec_energy*self.filt_array, axis=0)
                          / np.sum(filt_weights, axis=0))

    def _veldisp_spec(self):
        """ Convolve spectrum with a Gaussian in velocity space to model
        velicoty dispersion. """
        redshift = self.model_comp["redshift"]

        mask = ((self.chosen_wavs > self.spec_wavs[0]/(1+redshift))
                & (self.chosen_wavs < self.spec_wavs[-1]/(1+redshift)))

        specslice = self.spectrum_full[mask]

        vres = 3*10**5/np.max(self.R)/2.
        sigma_pix = self.model_comp["veldisp"]/vres
        x_kernel_pix = np.arange(-4*int(sigma_pix+1),
                                 4*int(sigma_pix+1)+1, 1.)

        kernel = ((1./np.sqrt(2*np.pi)/sigma_pix)
                  * np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2)))

        self.spectrum_full_veldisp = np.convolve(self.spectrum_full, kernel,
                                                 mode="valid")

        # Resample onto the requested wavelength grid
        ker_size = x_kernel_pix.shape[0]

        if ker_size > 1:
            old_wavs_rest = self.chosen_wavs[(ker_size-1)//2:-(ker_size-1)//2]

            spec_fluxes = np.interp(self.spec_wavs, old_wavs_rest*(1+redshift),
                                    self.spectrum_full_veldisp,
                                    left=0, right=0)

        else:
            spec_fluxes = np.interp(self.spec_wavs,
                                    self.chosen_wavs*(1+redshift),
                                    self.spectrum_full_veldisp,
                                    left=0, right=0)

        return np.array([self.spec_wavs, spec_fluxes]).T

    def update(self, model_components):
        """ Updates the model to the new numerical values specified in 
        the new model_components dictionary. Adding/removing components
        and changing non-numerical values is not supported. """

        self.model_comp = model_components

        # sfh: Star formation history object
        self.sfh = star_formation_history(self.model_comp)

        # self.spectrum_full: full spectrum sampled on chosen_wavs
        self.spectrum_full = np.zeros(self.chosen_wavs.shape[0])

        # separate_lines: independent record of line fluxes
        separate_lines = np.zeros(utils.line_wavs.shape[0])

        # dust_flux: total flux absorbed by dust (erg/s).
        dust_flux = 0.

        # The CF00 dust model has an extra free parameter n which raises
        # k_cont to a power, this power is set to 1 for all other types.
        if self.dust_on:
            if self.model_comp["dust"]["type"] == "CF00":
                n = -self.model_comp["dust"]["n"]

            else:
                n = 1.

        # Loop over all star formation history components of the model.
        for name in self.sfh_components:

            # Get relevant star-formation and chemical-enrichment info.
            sfh_weights = np.expand_dims(self.sfh.weights[name], axis=1)
            zmet_weights = self.sfh.ceh.zmet_weights[name]

            # Interpolate stellar grids in logU and metallicity.
            grid = self._combine_stellar_grids(zmet_weights)

            if self.nebular_on:
                nebular_grid, line_grid = self._combine_neb_grids(zmet_weights)

            # nlines: no of ages affected by nebular emission
            # frac_bc: affected fraction of the final affected age
            if "t_bc" in list(self.model_comp):
                t_bc = self.model_comp["t_bc"]*10**9

                age_mask = (utils.chosen_age_lhs < t_bc)
                nlines = utils.chosen_age_lhs[age_mask].shape[0]
                frac_bc = ((t_bc - utils.chosen_age_lhs[nlines-1])
                           / utils.chosen_age_widths[nlines-1])

            else:
                if self.nebular_on or self.extra_dust_on:
                    sys.exit("Bagpipes: t_bc is required if nebular emission"
                             + " or more dust for young ages are specified.")

            if self.nebular_on:
                # Add the nebular grid to the stellar grid
                grid[:nlines-1, :] += nebular_grid[:nlines - 1, :]
                grid[nlines-1, :] += frac_bc*nebular_grid[nlines - 1, :]

                # collapse line_grid over the age axis weighted by sfh
                line_grid[nlines-1, :] *= frac_bc
                lines = np.sum(sfh_weights[:nlines]*line_grid[:nlines, :],
                               axis=0)

            # Add extra dust on the youngest ages in the grid.
            if self.dust_on and self.extra_dust_on:
                eta = self.model_comp["dust"]["eta"]
                Av = self.model_comp["dust"]["Av"]

                extra_dust = 10**(-(eta - 1)*Av*self.k_cont**n/2.5)

                # Add dust to the grid
                grid_no_dust = np.copy(grid[:nlines, :])
                grid[:nlines-1, :] *= extra_dust
                grid[nlines-1, :] *= extra_dust*frac_bc + (1. - frac_bc)

                absorbed_spec = np.sum(grid_no_dust - grid[:nlines, :], axis=0)
                dust_flux += np.trapz(absorbed_spec, x=self.chosen_wavs)

                if self.nebular_on:
                    # Add extra dust to the separate lines.
                    lines *= 10**(-(eta - 1)*Av*self.k_line**n/2.5)

            # Collapse the grid along the age axis weighted by the SFH.
            self.spectrum_full += np.sum(sfh_weights*grid, axis=0)

            if self.nebular_on:
                separate_lines += lines

        # Hydrogen ionizing continuum is removed here by default.
        if self.keep_ionizing is not True:
            self.spectrum_full[self.chosen_wavs < 911.8] = 0.

            if self.nebular_on:
                separate_lines[utils.line_wavs < 911.8] = 0.

        # Apply diffuse dust absorption to the model
        if self.dust_on:
            Av = self.model_comp["dust"]["Av"]
            self.spectrum_full_nodust = np.copy(self.spectrum_full)
            self.spectrum_full *= 10**(-Av*self.k_cont**n/2.5)

            absorbed_spectrum = self.spectrum_full_nodust - self.spectrum_full
            dust_flux += np.trapz(absorbed_spectrum, x=self.chosen_wavs)

            if self.nebular_on:
                separate_lines *= 10**(-Av*self.k_line**n/2.5)

        # Apply intergalactic medium absorption to the model
        redshift = self.model_comp["redshift"]

        if redshift > 0.:
            mask = (utils.igm_redshifts < redshift)
            igm_ind = utils.igm_redshifts[mask].shape[0]

            if igm_ind == utils.igm_redshifts.shape[0]:
                igm_ind -= 1
                hi_zred_factor = 1.
                low_zred_factor = 0.

            else:
                width = (utils.igm_redshifts[igm_ind]
                         - utils.igm_redshifts[igm_ind-1])

                igm_z = utils.igm_redshifts[igm_ind]
                low_zred_factor = (igm_z - redshift)/width
                hi_zred_factor = 1. - low_zred_factor

            # interpolate igm transmission from pre-loaded grid
            igm_trans = (self.igm_cont[igm_ind-1, :]*low_zred_factor
                         + self.igm_cont[igm_ind, :]*hi_zred_factor)

            self.spectrum_full[self.igm_mask] *= igm_trans

            if self.nebular_on:
                igm_trans_lines = (self.igm_lines[igm_ind-1, :]*low_zred_factor
                                   + self.igm_lines[igm_ind, :]*hi_zred_factor)

                separate_lines *= igm_trans_lines

        # Add dust emission if dust keyword "temp" is specified.
        if self.dust_on and "temp" in list(self.model_comp["dust"]):

            temp = self.model_comp["dust"]["temp"]

            if "beta" in list(self.model_comp["dust"]):
                beta = self.model_comp["dust"]["beta"]

            else:
                beta = 1.5

            # At certain points in parameter space, this factor gets too
            # big to take the exponential of, resulting in math errors.
            hc_div_k = (6.626*10**-34)*(3*10**8)/(1.38*10**-23)
            exponent = hc_div_k/temp/(self.chosen_wavs*10**-10)

            gb_fnu = np.zeros_like(self.chosen_wavs)

            # Therefore only actually evaluate the exponential if
            # greybody_factor is less than 300, otherwise greybody flux
            # is zero.

            mask = (exponent < 300.)
            c_div_wavs = (3*10**8)/(self.chosen_wavs[mask]*10**-10)
            gb_fnu[mask] = c_div_wavs**(3+beta)/(np.exp(exponent[mask]) - 1.)

            # convert to f_lambda
            dust_emission = gb_fnu*(self.chosen_wavs*10**-10)**2 
            # normailse 
            dust_emission /= np.trapz(dust_emission, x=self.chosen_wavs)
            dust_emission *= dust_flux

            self.spectrum_full += dust_emission

        # Redshift the model spectrum and put it into erg/s/cm2/A unless
        # z = 0, in which case final units are erg/s/A.
        if redshift != 0.:

            # Unit conversion is Mpc to cm.
            ldist_cm = 3.086*10**24*np.interp(redshift, utils.z_array,
                                              utils.ldist_at_z,
                                              left=0, right=0)

            # convert to flux at given redshift in L_sol/s/A/cm^2.
            self.spectrum_full /= (4*np.pi*ldist_cm**2)

            # reduce flux by a factor of 1/(1+z) due to redshifting.
            self.spectrum_full /= (1+redshift)

            if self.nebular_on:
                separate_lines /= (4*np.pi*(ldist_cm)**2)

        # convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.
        self.spectrum_full *= 3.826*10**33

        if self.nebular_on:
            separate_lines *= 3.826*10**33

        # Resample filter profiles onto redshifted model wavelengths
        if self.filt_list is not None:
            self.photometry = self._calculate_photometry()

        # Generate the output spectrum if requested
        if self.spec_wavs is not None:

            if "veldisp" in list(self.model_comp):
                self.spectrum = self._veldisp_spec()

            # Otherwise just resample onto the requested wavelength grid
            else:
                spec_fluxes = np.interp(self.spec_wavs,
                                        self.chosen_wavs*(1+redshift),
                                        self.spectrum_full,
                                        left=0, right=0)

                self.spectrum = np.array([self.spec_wavs, spec_fluxes]).T

        # Apply spectral polynomial if requested
        if (("polynomial" in list(self.model_comp))
                and (self.spec_wavs is not None)):

            poly_coefs = []

            poly_dict = self.model_comp["polynomial"]

            while str(len(poly_coefs)) in list(poly_dict):
                poly_coefs.append(poly_dict[str(len(poly_coefs))])

            points = ((self.spec_wavs - self.spec_wavs[0])
                      / (self.spec_wavs[-1] - self.spec_wavs[0]))

            self.polynomial = cheb(points, poly_coefs)

            self.spectrum[:, 1] /= self.polynomial

        else:
            self.polynomial = None

        # line_fluxes: dict of output line fluxes in erg/s/cm^2 or erg/s
        # at redshift zero.
        if self.nebular_on:
            self.line_fluxes = dict(zip(utils.line_names, separate_lines))

        # Optionally switch spectrum/photometry units to microjanskys.
        const = 10**-29*2.9979*10**18

        if self.spec_units == "mujy" and self.spec_wavs is not None:
            self.spectrum[:, 1] /= ((const/self.spectrum[:, 0]
                                     / self.spectrum[:, 0]))

        if self.phot_units == "mujy" and self.filt_list is not None:
            self.photometry /= (const/self.eff_wavs/self.eff_wavs)

    def get_restframe_UVJ(self):
        """ Return rest-frame UVJ magnitudes for this model. """

        if self.UVJ_filt_names is None:

            filt_list_path = utils.install_dir + "/pipes_filters/UVJ.filt_list"
            self.UVJ_filt_names = np.loadtxt(filt_list_path, dtype="str")

            self.UVJ_filt_array = np.zeros((self.chosen_wavs.shape[0], 3))

            for i in range(len(self.UVJ_filt_names)):
                filt_path = (utils.install_dir + "/pipes_filters/"
                             + self.UVJ_filt_names[i])

                filter_raw = np.loadtxt(filt_path, usecols=(0, 1))

                self.UVJ_filt_array[:, i] = np.interp(self.chosen_wavs,
                                                      filter_raw[:, 0],
                                                      filter_raw[:, 1],
                                                      left=0, right=0)

            ABzpt_fluxes = (3.*10**18*3631.*10**-23)/(self.chosen_wavs**2)

            spec_energy = np.expand_dims(ABzpt_fluxes*self.wav_widths, axis=1)

            filt_weights = self.UVJ_filt_array*np.expand_dims(self.wav_widths,
                                                              axis=1)

            self.ABzpt = np.sum(spec_energy*self.UVJ_filt_array, axis=0)
            self.ABzpt /= np.sum(filt_weights, axis=0)

        spec_energy = np.expand_dims(self.spectrum_full*self.wav_widths,
                                     axis=1)

        filt_weights = self.UVJ_filt_array*np.expand_dims(self.wav_widths,
                                                          axis=1)

        UVJ_flux = np.sum(spec_energy*self.UVJ_filt_array, axis=0)
        UVJ_flux /= np.sum(filt_weights, axis=0)

        UVJ_ABmag = -2.5*np.log10(np.squeeze(UVJ_flux/self.ABzpt))

        return UVJ_ABmag

    def plot(self, show=True):
        """ Make a quick plot of the model spectral outputs. """

        return plotting.plot_model_galaxy(self, show=True)

from __future__ import print_function, division, absolute_import

import numpy as np 
import sys

from numpy.polynomial.chebyshev import chebval as cheb

from . import utils
from .star_formation_history import *
from .chemical_evolution_history import *
from . import plotting

# Ignore division by zero and overflow warnings
np.seterr(divide='ignore', invalid='ignore', over="ignore")

class Model_Galaxy:
    """ Build a model galaxy spectrum. Note, at least one of filtlist
    or spec_wavs must be defined.

    Parameters
    ----------

    model_components : dict
        A dictionary containing information about the model you wish to
        generate. 

    filtlist : str - optional
        The name of the filtlist: a collection of filter files through
        which photometric fluxes will be calculated.

    spec_wavs : array - optional
        An array of wavelengths at which spectral fluxes should be
        returned.

    spec_units : str - optional
        The units the output spectrum and photometry will be returned
        in. Default is "ergscma" for ergs per second per centimetre
        squared per angstrom, can be set to "mujy" for microjanskys.
    
    phot_units : str - optional
        The units the output spectrum and photometry will be returned
        in. Default is "mujy" for microjanskys, can be set to "ergscma"
        for ergs per second per centimetre squared per angstrom.    
    """


    def __init__(self, model_comp, filtlist=None, spec_wavs=None,
                 spec_units="ergscma", phot_units="ergscma"):

        if spec_wavs is None and filtlist is None:
            sys.exit("Bagpipes: Specify either filtlist or spec_wavs.")

        self.model_comp = model_comp
        self.filtlist = filtlist
        self.spec_wavs = spec_wavs
        self.spec_units = spec_units
        self.phot_units = phot_units

        self.dust_on = ("dust" in list(self.model_comp))
        self.nebular_on = ("nebular" in list(self.model_comp))

        # modelgrids: stores stellar grids resampled onto chosen_wavs
        self.stellar_grids = {}

        # cloudygrids: stores nebular grids resampled onto chosen_wavs
        self.nebular_grids = {}

        # UVJ_filt_names: UVJ filter names, see get_restframe_UVJ.
        self.UVJ_filt_names = None

        # Loads filter curves from filtlist and finds max and min wavs.
        if self.filtlist is not None:
            self._load_filter_curves()

        # Calculate chosen_wavs, a common wavelength sampling.
        self._get_chosen_wavs()

        # Resample filter curves to chosen_wavs.
        if self.filtlist is not None:
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
        for key in list(self.model_comp):
            if (key in component_types or key[:-1] in component_types):
                self.sfh_components.append(key)

        # Check the model has at least one SFH component.
        if len(self.sfh_components) == 0:
            sys.exit("Bagpipes: No SFH components passed to Model_Galaxy.")

        self.update(self.model_comp)


    """ Loads filter files for the specified filtlist, truncates zeros
    from their edges and calculates effective wavelength values which 
    are added to eff_wavs. """
    def _load_filter_curves(self):

        # filterlist: a list of the filter file names associated with 
        # the specified filtlist.
        self.filt_names = np.loadtxt(working_dir + "/pipes/filters/"
                                     + self.filtlist + ".filtlist", 
                                     dtype="str")

        # self.filt_dict: a dict containing the raw filter files
        self.filt_dict = {}
        for filt in self.filt_names:
            self.filt_dict[filt] = np.loadtxt(working_dir
                                              + "/pipes/filters/"
                                              + filt, 
                                              usecols=(0, 1))

            #Get rid of trailing zeros at ends of the filter files
            while self.filt_dict[filt][0, 1] == 0.:
                self.filt_dict[filt] = self.filt_dict[filt][1:, :]

            while self.filt_dict[filt][-1, 1] == 0.:
                self.filt_dict[filt] = self.filt_dict[filt][:-1, :]

        # min_phot_wav, max_phot_wav: the min and max wavelengths
        # covered by any of the photometric bands.
        self.min_phot_wav = 9.9*10**99
        self.max_phot_wav = 0.

        for filt in self.filt_names:
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
        self.eff_wavs = np.zeros(len(self.filt_names))

        for i in range(len(self.filt_names)):
            filt = self.filt_names[i]
            dlambda = make_bins(self.filt_dict[filt][:, 0])[1]
            filt_weights = dlambda*self.filt_dict[filt][:, 1]
            self.eff_wavs[i] = np.sqrt(np.sum(filt_weights)
                                       /np.sum(filt_weights
                                       /self.filt_dict[filt][:, 0]**2))


    """ Resamples the filter curves onto the chosen_wavs and creates a
    2D grid filt_array of all filter curves. """
    def _resample_filter_curves(self):

        # filt_rest: An array to contain the rest frame sampled filter
        # profiles used to generate the model photometry.
        self.filt_rest = np.zeros((self.chosen_wavs.shape[0],
                                              len(self.filt_names)))

        for i in range(len(self.filt_names)):
            filt = self.filt_names[i]
            self.filt_rest[:,i] = np.interp(self.chosen_wavs, 
                                            self.filt_dict[filt][:, 0], 
                                            self.filt_dict[filt][:, 1], 
                                            left=0, right=0)

        # filt_array: An array to contain the resampled filter profiles
        # used to generate the model photometry.
        self.filt_array = np.zeros((self.chosen_wavs.shape[0], 
                                    len(self.filt_names)))

        # model_widths: the widths of the bins with midp chosen_wavs.
        model_widths = make_bins(self.chosen_wavs)[1]

        # wav_widths: An array containing the relative widths*wavs
        # for each point in the spectrum.
        self.wav_widths = model_widths*self.chosen_wavs


    """ Sets up chosen_wavs: sets up an optimal wavelength sampling for
    the models within the class to be interpolated onto to maximise
    the speed of the code. """
    def _get_chosen_wavs(self):
        x = [1.]

        if self.spec_wavs is None:
            self.max_wavs = [self.min_phot_wav/(1.+max_zred), 
                             1.01*self.max_phot_wav, 10**8]

            self.R = [10., 100., 10.]

        elif self.filtlist is None:
            self.max_wavs = [self.spec_wavs[0]/(1.+max_zred), 
                             self.spec_wavs[-1], 10**8]

            self.R = [10., 600., 10.]

        else:
            if (self.spec_wavs[0] > self.min_phot_wav
                    and self.spec_wavs[-1] < self.max_phot_wav):

                self.max_wavs = [self.min_phot_wav/(1.+max_zred), 
                                 self.spec_wavs[0]/(1.+max_zred), 
                                 self.spec_wavs[-1], 
                                 self.max_phot_wav, 10**8]

                self.R = [10., 100., 600., 100., 10.]

            elif (self.spec_wavs[0] < self.min_phot_wav 
                      and self.spec_wavs[-1] < self.max_phot_wav):

                self.max_wavs = [self.spec_wavs[0]/(1.+max_zred), 
                                 self.spec_wavs[-1], 
                                 self.max_phot_wav, 10**8]

                self.R = [10., 600., 100., 10.]

            if (self.spec_wavs[0] > self.min_phot_wav 
                    and self.spec_wavs[-1] > self.max_phot_wav):

                self.max_wavs = [self.min_phot_wav/(1.+max_zred), 
                                 self.spec_wavs[0]/(1.+max_zred), 
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


    """ Set up the igm grids on the same wavelength sampling as the 
    nebular and stellar models. """
    def _resample_igm_grids(self):

        igm_wav_mask = (self.chosen_wavs > 911.8) & (self.chosen_wavs < 1220.)
        
        # set up igm_cont to contain transmission values for continuum
        # from 912A to 1216A as a function of redshift.
        self.igm_cont = np.zeros((utils.igm_redshifts.shape[0], 
                                  self.chosen_wavs[igm_wav_mask].shape[0]))
        
        # Resample from the raw grids held in the utils file.
        for i in range(igm_redshifts.shape[0]):
            self.igm_cont[i, :] = np.interp(self.chosen_wavs[igm_wav_mask],
                                            utils.igm_wavs,
                                            utils.igm_grid[i, :])

        # Set up igm_lines to contain a grid of transmission values for 
        # emission lines from 912A to 1216A as a function of redshift.
        self.igm_lines = np.zeros((igm_redshifts.shape[0], 
                                   utils.line_wavs.shape[0]))
        
        for i in range(igm_redshifts.shape[0]):
            self.igm_lines[i, :] = np.interp(utils.line_wavs, utils.igm_wavs,
                                             utils.igm_grid[i, :],
                                             left=0., right=1.)
                                          

    """ Returns k values for chosen dust curve normalised to Av = 1. """
    def _get_dust_model(self, wavs):

        dust_type = self.model_comp["dust"]["type"]

        if dust_type == "Calzetti":
            k_lam = np.loadtxt(install_dir + "/tables/dust/Calzetti_2000.txt")

            return np.interp(wavs, k_lam[:, 0], k_lam[:, 1], right=0)

        elif dust_type == "Cardelli":
            k_lam = np.loadtxt(install_dir + "/tables/dust/Cardelli_1989.txt")

            return np.interp(wavs, k_lam[:, 0], k_lam[:, 1], right=0)

        elif dust_type == "CF00":
            return wavs/5500.

        else:
            sys.exit("BAGPIPES: Dust type not recognised.")


    """ Loads stellar grid at specified metallicity, and resamples it to 
    chosen_wavs. """
    def _load_stellar_grid(self, zmet_ind):

        grid = load_stellar_grid(zmet_ind)

        # Resample model grid onto chosen_wavs.
        grid_res = np.zeros((grid.shape[0], self.chosen_wavs.shape[0]))

        for i in range(grid_res.shape[0]):
            grid_res[i, :] = np.interp(self.chosen_wavs,
                                       utils.gridwavs[utils.model_type],
                                       grid[i, :], left=0, right=0)

        # Add the resampled grid to the modelgrids dictionary
        self.stellar_grids[str(zmet_ind)] = grid_res


    """ Loads Cloudy nebular continuum and emission lines at specified
    metallicity and logU. """
    def _load_cloudy_grid(self, zmet_val, logU):

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
                cont_grid[:,ind] += raw_line_grid[:,i]/width
        
        # Add the finished cloudy grid to the cloudygrids dictionary
        self.nebular_grids[str(zmet_val) + str(logU)] = cont_grid


    """ Combines nebular and stellar grids in metallicity and logU to 
    create grids which are ready to be multiplied by the sfh. """
    def _combine_grids(self, zmet_weights, logU_vals, logU_weights):

        # Check relevant stellar and nebular grids have been loaded.
        zmet_vals = utils.zmet_vals[utils.model_type]

        for zmet_val in zmet_vals[zmet_weights != 0.]:
            if zmet_val not in list(self.stellar_grids):
                self._load_stellar_grid(zmet_val)
            
            if self.nebular_on:
                for logU_val in (logU_vals[1], logU_vals[0]):
                    key = str(zmet_val) + str(logU_val)
                    if key not in list(self.nebular_grids):
                        self._load_cloudy_grid(zmet_val, logU_val)

        # Create blank arrays for stellar and nebular grids.
        stellar_key = list(self.stellar_grids)[0]
        nebular_key = list(self.nebular_grids)[0]
        line_key = list(utils.allcloudylinegrids)[0]
        stellar_grid = np.zeros_like(self.stellar_grids[stellar_key])
        nebular_grid = np.zeros_like(self.nebular_grids[nebular_key])
        line_grid = np.zeros_like(utils.allcloudylinegrids[line_key])

        linegrids = utils.allcloudylinegrids

        # Combine by metallicity and logU to get age vs wavelength grids
        for zmet_val in zmet_vals[zmet_weights != 0.]:
            wt = zmet_weights[zmet_vals == zmet_val]
            stellar_grid += wt*self.stellar_grids[str(zmet_val)]

            if self.nebular_on:
                high_key = str(zmet_val) + str(logU_vals[1])
                low_key = str(zmet_val) + str(logU_vals[0])

                nebular_grid += wt*(logU_weights[1]*self.nebular_grids[high_key]
                                    + logU_weights[0]*self.nebular_grids[low_key])
                
                line_grid += wt*(logU_weights[1]*linegrids[high_key]
                                 + logU_weights[0]*linegrids[low_key])

        return stellar_grid, nebular_grid, line_grid


    """ Updates the model with new attributes passed in the model_comp
    dictionary. """
    def update(self, model_comp):

        self.model_comp = model_comp

        # sfh: Star formation history object
        self.sfh = star_formation_history(self.model_comp)

        # ceh: Chemical evolution history object
        self.ceh = chemical_evolution_history(self.model_comp)

        self.living_stellar_mass = {}
        self.living_stellar_mass["total"] = 0.

        # self.spectrum_full: 1d spectrum sampled on chosen_wavs
        self.spectrum_full = np.zeros(self.chosen_wavs.shape[0])

        # composite_lines: will contain the line strengths obtained from
        # adding contributions from all SFH components together.
        composite_lines = np.zeros(utils.line_wavs.shape[0])

        # total_dust_flux: saves the total flux absorbed by dust to set
        # the normalisation of the dust emission component (erg/s).
        total_dust_flux = 0.

        # Figure out which are the nebular grids in logU closest to the
        # chosen logU and what fraction of each grid should be taken.
        if self.nebular_on:
            logU = self.model_comp["nebular"]["logU"]
            logU_vals = [np.min(utils.logU_grid[logU_grid >= logU]), 
                         np.max(utils.logU_grid[logU_grid < logU])]

            logU_weights = [(logU - logU_vals[0])/(logU_vals[1] - logU_vals[0]),
                            0.]

            logU_weights[1] = 1. - logU_weights[1]

        # The CF00 dust model has an extra free parameter n which raises
        # k_cont to a power, this power is set to 1 for all other types.
        if self.dust_on:
            if self.model_comp["dust"]["type"] == "CF00":
                n = -self.model_comp["dust"]["n"]

            else:
                n = 1.

        # Loop over all star formation history components of the model
        for comp in self.sfh_components:

            # Access relevant star-formation and chemical-enrichment
            sfh_weights = self.sfh.weight_widths[comp]
            zmet_weights = self.ceh.zmet_weights[comp]

            grids = self._combine_grids(zmet_weights, logU_vals, logU_weights)
            stellar_grid, cont_grid, line_grid = grids

            ##################
            #Calculate living stellar mass contribution from this SFH component
            comp_living_mass = np.sum(sfh_weights*np.sum(np.expand_dims(zmet_weights, 1)*chosen_live_frac, axis=0))
            self.living_stellar_mass["total"] += comp_living_mass
            self.living_stellar_mass[comp] = np.copy(comp_living_mass)
            
            # Calculate how many lines of the grids are affected by birth clouds and what fraction of the final line is affected
            # Otherwise check nothing which required t_bc to be specified has been specified and crash the code if so
            if "t_bc" in list(self.model_comp):
                nlines = chosen_age_lhs[chosen_age_lhs < self.model_comp["t_bc"]*10**9].shape[0]
                frac_bc = (self.model_comp["t_bc"]*10**9 - chosen_age_lhs[nlines-1])/chosen_age_widths[nlines-1]

            else:
                if self.nebular_on or self.dust_on and "eta" in list(self.model_comp["dust"]):
                    sys.exit("Bagpipes: t_bc must be specified if nebular emission or more dust for young ages are specified.")


            # This adds the nebular grids to the stellar grids
            if self.nebular_on:

                # Add nebular grid to stellar grid
                stellar_grid[:nlines-1, :] += cont_grid[:nlines-1, :]
                stellar_grid[nlines-1, :] += cont_grid[nlines-1, :]*frac_bc

                line_grid[nlines-1, :] *= frac_bc

                interpolated_cloudy_lines = np.sum(np.expand_dims(sfh_weights[:nlines], axis=1)*line_grid[:nlines, :], axis=0)


            # If extra dust on the youngest stellar component has been requested, this section adds it in before the grid is converted from SSPs into a CSP
            if self.dust_on and "eta" in list(self.model_comp["dust"]):

                T_dust = 10**(-(self.model_comp["dust"]["eta"] - 1)*self.model_comp["dust"]["Av"]*self.k_cont**n/2.5)

                stellar_grid_no_dust = np.copy(stellar_grid[:nlines, :])
                stellar_grid[:nlines-1, :] *= T_dust
                stellar_grid[nlines-1, :] *= T_dust*frac_bc + (1. - frac_bc)

                total_dust_flux += np.trapz(np.sum(stellar_grid_no_dust - stellar_grid[:nlines, :], axis=0), x=self.chosen_wavs)

                if self.nebular_on:
                    interpolated_cloudy_lines *= 10**(-(self.model_comp["dust"]["eta"] - 1)*self.model_comp["dust"]["Av"]*self.k_line**n/2.5)

            # Obtain CSP by performing a weighed sum over the SSPs and add it to the composite spectrum
            self.spectrum_full += np.sum(np.expand_dims(sfh_weights, axis=1)*stellar_grid, axis=0)

            if self.nebular_on:
                composite_lines += interpolated_cloudy_lines

            # The Hydrogen ionizing continuum is removed here by default
            if self.keep_ionizing is not True:
                self.spectrum_full[self.chosen_wavs < 911.8] = 0.

                if self.nebular_on:
                    composite_lines[utils.line_wavs < 911.8] = 0.

        # Apply diffuse dust absorption correction to the composite spectrum
        if self.dust_on:

            self.spectrum_full_nodust = np.copy(self.spectrum_full)
            self.spectrum_full *= 10**(-self.model_comp["dust"]["Av"]*self.k_cont**n/2.5)
            total_dust_flux += np.trapz(self.spectrum_full_nodust - self.spectrum_full, x = self.chosen_wavs)

            if self.nebular_on:
                composite_lines *= 10**(-self.model_comp["dust"]["Av"]*self.k_line**n/2.5)


        # Apply intergalactic medium absorption to the model spectrum
        if self.model_comp["redshift"] > 0.:
            zred_ind = igm_redshifts[igm_redshifts < self.model_comp["redshift"]].shape[0]

            high_zred_factor = (igm_redshifts[zred_ind] - self.model_comp["redshift"])/(igm_redshifts[zred_ind] - igm_redshifts[zred_ind-1])
            low_zred_factor = 1. - high_zred_factor

            # np.interpolate igm transmission from pre-loaded grid
            igm_trans = self.igm_cont[zred_ind-1, :]*low_zred_factor + self.igm_cont[zred_ind, :]*high_zred_factor
            
            self.spectrum_full[(self.chosen_wavs < 1220.) & (self.chosen_wavs > 911.8)] *= igm_trans

            if self.nebular_on:
                igm_trans_lines = self.igm_lines[zred_ind-1, :]*low_zred_factor + self.igm_lines[zred_ind, :]*high_zred_factor

                composite_lines *= igm_trans_lines
        

        # Add dust emission model if dust keyword "temp" is specified
        if self.dust_on and "temp" in list(self.model_comp["dust"]):

            wavs = self.chosen_wavs*10**-10 # Convert wavelength values from microns to metres

            if "beta" in list(self.model_comp["dust"]):
                beta = self.model_comp["dust"]["beta"]

            else:
                beta = 1.5

            # At certain points in parameter space, this factor gets too big to take the exponential of, resulting in math errors
            greybody_factor = (6.626*10**-34)*(3*10**8)/(1.38*10**-23)/self.model_comp["dust"]["temp"]/wavs

            S_jy_greybody = np.zeros(wavs.shape[0])

            # Therefore only actually evaluate the exponential if greybody_factor is greater than 300, otherwise greybody flux is zero
            S_jy_greybody[greybody_factor < 300.] = (((3*10**8)/wavs[greybody_factor < 300.])**(3+beta))/(np.exp(greybody_factor[greybody_factor < 300.]) - 1.)

            dust_emission = S_jy_greybody*wavs**2 #convert to f_lambda
            dust_emission /= np.trapz(dust_emission, x=self.chosen_wavs)
            dust_emission *= total_dust_flux

            self.spectrum_full += dust_emission
            
        # Redshift the model spectrum and put it into erg/s/cm2/A unless z = 0, in which case final units are erg/s/A
        if self.model_comp["redshift"] != 0.:
            self.spectrum_full /= (4*np.pi*(np.interp(self.model_comp["redshift"], z_array, ldist_at_z, left=0, right=0)*3.086*10**24)**2) #convert to observed flux at given redshift in L_sol/s/A/cm^2
            self.spectrum_full /= (1+self.model_comp["redshift"]) #reduce flux by a factor of 1/(1+z) to account for redshifting

            if self.nebular_on:
                composite_lines /= (4*np.pi*(np.interp(self.model_comp["redshift"], z_array, ldist_at_z, left=0, right=0)*3.086*10**24)**2) #convert to observed flux at given redshift in L_sol/s/A/cm^2

        self.spectrum_full *= 3.826*10**33 #convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.

        if self.nebular_on:
            composite_lines *= 3.826*10**33 #convert to erg/s/cm^2, or erg/s if redshift = 0.

        if self.filtlist is not None:
            # Resample the filter profiles onto the redshifted model wavelength grid
            for i in range(len(self.filt_names)):
                self.filt_array[:,i] = np.interp(self.chosen_wavs*(1.+self.model_comp["redshift"]), self.chosen_wavs, self.filt_rest[:,i], left=0, right=0)

            # photometry: The flux values for the model spectrum observed through the filters specified in filtlist
            self.photometry = np.squeeze(np.sum(np.expand_dims(self.spectrum_full*self.wav_widths, axis=1)*self.filt_array, axis=0)/np.sum(self.filt_array*np.expand_dims(self.wav_widths, axis=1), axis=0))
            """ Output photometry array, contains a column of flux values, by default in erg/s/cm^2/A or erg/s/A at redshift zero. """

        # Generate the output spectrum if requested
        if self.spec_wavs is not None:

            # Apply velocity dispersion to the output spectrum if requested
            if "veldisp" in list(self.model_comp):

                specslice = self.spectrum_full[(self.chosen_wavs > self.spec_wavs[0]/(1+self.model_comp["redshift"])) & (self.chosen_wavs < self.spec_wavs[-1]/(1+self.model_comp["redshift"]))]

                vres = 3*10**5/np.max(self.R)/2.
                sigma_pix = self.model_comp["veldisp"]/vres
                x_kernel_pix = np.arange(-4*int(sigma_pix+1), 4*int(sigma_pix+1)+1, 1.)
                kernel = (1./np.sqrt(2*np.pi)/sigma_pix)*np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
                self.spectrum_full_veldisp = np.convolve(self.spectrum_full, kernel, mode="valid")

                # Resample onto the requested wavelength grid
                if x_kernel_pix.shape[0] > 1:
                    self.spectrum = np.array([self.spec_wavs, np.interp(self.spec_wavs, self.chosen_wavs[(x_kernel_pix.shape[0]-1)//2:-(x_kernel_pix.shape[0]-1)//2]*(1+self.model_comp["redshift"]), self.spectrum_full_veldisp, left=0, right=0)]).T
                    """ Output spectrum array, contains a column of wavelengths in Angstroms and a column of flux values, by default in erg/s/cm^2/A or erg/s/A at redshift zero. """

                else:
                    self.spectrum = np.array([self.spec_wavs, np.interp(self.spec_wavs, self.chosen_wavs*(1+self.model_comp["redshift"]), self.spectrum_full_veldisp, left=0, right=0)]).T

            # Otherwise just resample onto the requested wavelength grid
            else:
                self.spectrum = np.array([self.spec_wavs, np.interp(self.spec_wavs, self.chosen_wavs*(1+self.model_comp["redshift"]), self.spectrum_full, left=0, right=0)]).T

        # Apply spectral polynomial if requested
        if "polynomial" in list(self.model_comp):

            polycoefs = []

            while str(len(polycoefs)) in list(self.model_comp["polynomial"]):
                polycoefs.append(self.model_comp["polynomial"][str(len(polycoefs))])

            points = (self.spec_wavs - self.spec_wavs[0])/(self.spec_wavs[-1] - self.spec_wavs[0])

            self.polynomial = cheb(points, polycoefs)

            self.spectrum[:, 1] *= self.polynomial

        else:
            self.polynomial = None

        if self.nebular_on:
            self.line_fluxes = dict(zip(utils.line_names, composite_lines))
            """ Dictionary of output emission line fluxes in erg/s/cm^2 or erg/s at redshift zero. """

        # Return output spectrum and photometry in microjanskys instead of erg/s/cm^2/A if requested.
        if self.spec_units == "mujy" and self.spec_wavs is not None:
            self.spectrum[:, 1] /= (10**-29)*(2.9979*10**18/self.spectrum[:, 0]/self.spectrum[:, 0])

        if self.phot_units == "mujy" and self.filtlist is not None:
            self.photometry /= (10**-29)*(2.9979*10**18/self.eff_wavs/self.eff_wavs)


    """ Return rest-frame UVJ magnitudes for this model. """
    def get_restframe_UVJ(self):

        if self.UVJ_filt_names is None:

            # filterlist: a list of the filter file names for UVJ.
            self.UVJ_filt_names = np.loadtxt(install_dir + "/filters/UVJ.filtlist", dtype="str")

            # filt_rest: An array to contain the rest frame sampled filter profiles used to generate the model photometry
            self.UVJ_filt_array = np.zeros((self.chosen_wavs.shape[0], 3))

            for i in range(len(self.UVJ_filt_names)):
                filter_raw = np.loadtxt(install_dir + "/filters/" + self.UVJ_filt_names[i], usecols=(0, 1))

                self.UVJ_filt_array[:,i] = np.np.interp(self.chosen_wavs, filter_raw[:, 0], filter_raw[:, 1], left=0, right=0)
            
            ABzpt_spec = np.array([self.chosen_wavs, (3.*10**18*3631.*10**-23)/(self.chosen_wavs**2)]).T

            self.ABzpt = np.sum(np.expand_dims(ABzpt_spec[:, 1]*self.wav_widths, axis=1)*self.UVJ_filt_array, axis=0)/np.sum(np.expand_dims(self.wav_widths, axis=1)*self.UVJ_filt_array, axis=0)

        UVJ = (np.squeeze(np.sum(np.expand_dims(self.spectrum_full*self.wav_widths, axis=1)*self.UVJ_filt_array, axis=0)/np.sum(self.UVJ_filt_array*np.expand_dims(self.wav_widths, axis=1), axis=0)))

        UVJ = -2.5*np.log10((np.squeeze(np.sum(np.expand_dims(self.spectrum_full*self.wav_widths, axis=1)*self.UVJ_filt_array, axis=0)/np.sum(self.UVJ_filt_array*np.expand_dims(self.wav_widths, axis=1), axis=0)))/self.ABzpt)

        return UVJ


    """ Plot the model. """
    def plot(self, show=True):
        return plotting.plot_model_galaxy(self, show=show)

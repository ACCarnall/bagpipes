from __future__ import print_function

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import sys
import os
import time

from numpy import interp
from numpy.polynomial.chebyshev import chebval as cheb

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.3)
z_array = np.arange(0., 10., 0.01)
ldist_at_z = cosmo.luminosity_distance(z_array).value

from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

import star_formation_history
import model_manager as models

if not os.path.exists(models.install_dir + "/tables/IGM/D_IGM_grid_Inoue14.txt"):
	sys.path.append(models.install_dir + "/tables/IGM")
	import IGM_Inoue2014 as igm 
	igm.make_table()

D_IGM_grid_global = np.loadtxt(models.install_dir + "/tables/IGM/D_IGM_grid_Inoue14.txt")
IGM_redshifts = np.arange(0.0, 10.01, 0.01)
IGM_wavs_global = np.arange(1.0, 1225.01, 1.0)


class Model_Galaxy:
	""" Build a model galaxy spectrum.

	Parameters
	----------

	model_components : dict
		A dictionary containing information about the model you wish to generate. 

	filtlist : str (optional)
		The name of the filtlist: a collection of filter files through which photometric fluxes will be calculated.

	output_specwavs : array (optional)
		An array of wavelengths at which spectral fluxes should be returned.

	out_units_spec : str (optional)
		The units the output spectrum and photometry will be returned in. Default is "ergscma" for ergs per second per centimetre squared per angstrom, can be set to "mujy" for microjanskys.
	
	out_units_phot : str (optional)
		The units the output spectrum and photometry will be returned in. Default is "mujy" for microjanskys, can be set to "ergscma" for ergs per second per centimetre squared per angstrom.
	
	"""

	def __init__(self, model_components, filtlist=None, output_specwavs=None, out_units_spec="ergscma", out_units_phot="ergscma"):

		if output_specwavs is None and filtlist is None:
			sys.exit("Bagpipes: Either a filtlist or output_specwavs must be specified")

		self.model_comp = model_components

		self.filtlist = filtlist

		self.out_units_spec = out_units_spec
		self.out_units_phot = out_units_phot

		self.output_specwavs = output_specwavs

		# original_modelgrid_wavs: The wavelength sampling the model grids will have when they are loaded from file.
		self.original_modelgrid_wavs = np.copy(models.gridwavs[models.model_type])

		# modelgrids: Dictionary used to store model grids which have been resampled onto the desired wavelength sampling
		self.modelgrids = {}

		# cloudygrids: Dictionary used to store cloudy grids which have been resampled onto the desired wavelength sampling and normalised
		self.cloudygrids = {}

		# cloudylinewavs: Wavelength values for the lines added to the model from the cloudy grids
		self.cloudylinewavs = None

		# UVJ_filterlist: The list of filters for calculating rest frame UVJ magnitudes
		self.UVJ_filterlist = None

		# cloudylinelabels: an array of line labels from cloudy, used to construct line strength dictionary
		self.cloudylinelabels = np.genfromtxt(models.install_dir + "/tables/nebular/cloudy_lines.txt", dtype="str", delimiter="\t")

		# component_types: List of all possible types of star formation history (SFH) components
		self.component_types = star_formation_history.component_types

		# sfh_components: List of the SFH components for this model
		self.sfh_components = []

		# Recognise only dictionaries which have names listed in self.component_types as being SFH components, add them to sfh_components list
		for comp in list(self.model_comp):
			if (comp in self.component_types or comp[:-1] in self.component_types) and isinstance(self.model_comp[comp], dict):
				self.sfh_components.append(comp)

		# Check the model has at least one SFH component.
		if len(self.sfh_components) == 0:
			sys.exit("Bagpipes: Error, Model_Galaxy was not passed any recognised star formation history components in model_components dict.")

		# zmet_vals: An array of metallicity values at which model grids are available for the chosen set of SPS models.
		self.zmet_vals = np.copy(models.zmet_vals[models.model_type])
		self.zmet_vals_highres = np.arange(0., 10., 0.01) + 0.005

		self.zmet_lims = np.zeros(self.zmet_vals.shape[0]+1)
		for i in range(1, self.zmet_lims.shape[0]-1):
			self.zmet_lims[i] = (self.zmet_vals[i] + self.zmet_vals[i-1])/2.

		self.zmet_lims[-1] = 10.

		self.polynomial = None
		# Polynomial correction which has been applied to the spectrum. 

		# hc_k: combination of physical constants used for calculating dust emission
		self.hc_k = (6.626*10**-34)*(3*10**8)/(1.38*10**-23)

		# k_lambda and k_lambda_lines: dictionaries to contain k_lambda values for different dust curves at spectral wavelengths or the wavelengths of emission lines respectively
		self.k_lambda = {}
		self.k_lambda_lines = {}

		# This bit of code sets up the calculation of model fluxes in photometric bands if a filtlist is specified
		if self.filtlist is not None:

			# filterlist: a list of the filter file names associated with the specified filtlist
			self.filterlist = np.loadtxt(models.working_dir + "/pipes/filters/" + self.filtlist + ".filtlist", dtype="str")

			# filter_raw_dict: a dict containing the raw filter files
			filter_raw_dict = {}
			for filtername in self.filterlist:
				filter_raw_dict[filtername] = np.loadtxt(models.working_dir + "/pipes/filters/" + filtername, usecols=(0, 1))

				#Get rid of trailing zeros at either end of the filter files
				while filter_raw_dict[filtername][0,1] == 0.:
					filter_raw_dict[filtername] = filter_raw_dict[filtername][1:, :]

				while filter_raw_dict[filtername][-1,1] == 0.:
					filter_raw_dict[filtername] = filter_raw_dict[filtername][:-1, :]

			# min_phot_wav, max_phot_wav: the minimum and maximum wavelengths covered by any of the photometric bands
			min_phot_wav = 9.9*10**99
			max_phot_wav = 0.

			for filtername in self.filterlist:
				min_wav = filter_raw_dict[filtername][0, 0] - 2*(filter_raw_dict[filtername][1, 0] - filter_raw_dict[filtername][0, 0])
				max_wav = filter_raw_dict[filtername][-1, 0] + 2*(filter_raw_dict[filtername][-1, 0] - filter_raw_dict[filtername][-2, 0])

				if min_wav < min_phot_wav:
					min_phot_wav = min_wav

				if max_wav > max_phot_wav:
					max_phot_wav = max_wav

		""" This bit of code sets up chosen_modelgrid_wavs: The wavelength sampling the model grids will be resampled onto for manipulation 
		by the code. Setting this up is quite complex, it depends on whether spectra, photometry or both are requested, and relative positions
		in wavelength x: the minimum (first) wavelength value in chosen_modelgrid_wavs self.R: the spectral resolution (lambda/deltalambda)
		for the spectrum to be sampled at over the various spectral regions self.max_wavs: the maximum wavelength values for each spectral
		region, after this point the code moves onto the next self.R value """

		x = [1.]

		if output_specwavs is None:
			self.max_wavs = [min_phot_wav/(1.+models.max_zred), 1.01*max_phot_wav, 10**8]
			self.R = [10., 100., 10.]#[10., 100., 10.]

		elif self.filtlist is None:
			self.max_wavs = [output_specwavs[0]/(1.+models.max_zred), output_specwavs[-1], 10**8]
			self.R = [10., 600., 10.]#[10., 600., 10.]

		else:
			if output_specwavs[0] > min_phot_wav and output_specwavs[-1] < max_phot_wav:
				self.max_wavs = [min_phot_wav/(1.+models.max_zred), output_specwavs[0]/(1.+models.max_zred), output_specwavs[-1], max_phot_wav, 10**8]
				self.R = [10., 100., 600., 100., 10.]

			elif output_specwavs[0] < min_phot_wav and output_specwavs[-1] < max_phot_wav:
				self.max_wavs = [output_specwavs[0]/(1.+models.max_zred), output_specwavs[-1], max_phot_wav, 10**8]
				self.R = [10., 600., 100., 10.]

			if output_specwavs[0] > min_phot_wav and output_specwavs[-1] > max_phot_wav:
				self.max_wavs = [min_phot_wav/(1.+models.max_zred), output_specwavs[0]/(1.+models.max_zred), output_specwavs[-1], 10**8]
				self.R = [10., 100., 600., 10.]

		# Generate the wavelength grid the models will be resampled onto. This runs from 1 to 10**8 Angstroms with variable sampling
		for i in range(len(self.R)):
			if i == len(self.R)-1 or self.R[i] > self.R[i+1]:
				while x[-1] < self.max_wavs[i]:
					x.append(x[-1]*(1.+0.5/self.R[i]))

			else:
				while x[-1]*(1.+0.5/self.R[i]) < self.max_wavs[i]:
					x.append(x[-1]*(1.+0.5/self.R[i]))

		self.chosen_modelgrid_wavs = np.array(x)

		# keep_ionizing_continuum: A flag which can be set to prevent the code removing all flux below 912A.
		if "keep_ionizing_continuum" in list(self.model_comp) and self.model_comp["keep_ionizing_continuum"] == True:
			self.keep_ionizing_continuum = True

		else:
			self.keep_ionizing_continuum = False

		# D_IGM_grid: a grid of correction factors for IGM attenuation as a function of wavelength and redshift taken from Inoue et al. (2014)
		self.D_IGM_grid = np.zeros((IGM_redshifts.shape[0], self.chosen_modelgrid_wavs[(self.chosen_modelgrid_wavs > 911.8) & (self.chosen_modelgrid_wavs < 1220.)].shape[0]))
		
		for i in range(IGM_redshifts.shape[0]):
			self.D_IGM_grid[i,:] = interp(self.chosen_modelgrid_wavs[(self.chosen_modelgrid_wavs > 911.8) & (self.chosen_modelgrid_wavs < 1220.)], IGM_wavs_global, D_IGM_grid_global[i,:])

		# If filtlist is not None, finish off necessary calculations for photometry calculation
		if self.filtlist is not None:
			# filt_array_restframe: An array to contain the rest frame sampled filter profiles used to generate the model photometry
			self.filt_array_restframe = np.zeros((self.chosen_modelgrid_wavs.shape[0], len(self.filterlist)))

			for i in range(len(self.filterlist)):
				self.filt_array_restframe[:,i] = interp(self.chosen_modelgrid_wavs, filter_raw_dict[self.filterlist[i]][:,0], filter_raw_dict[self.filterlist[i]][:,1], left=0, right=0)

			# filt_array: An array to contain the resampled filter profiles used to generate the model photometry
			self.filt_array = np.zeros((self.chosen_modelgrid_wavs.shape[0], len(self.filterlist)))

			# model_lhs, model_widths: the left hand side positions and widths respectively of the chosen_modelgrid_wavs
			res_len = self.chosen_modelgrid_wavs.shape[0]
			model_widths = np.zeros(res_len)
			model_lhs = np.zeros(res_len+1)
			model_lhs[0] = self.chosen_modelgrid_wavs[0] - (self.chosen_modelgrid_wavs[1]-self.chosen_modelgrid_wavs[0])/2
			model_widths[res_len-1] = (self.chosen_modelgrid_wavs[res_len-1] - self.chosen_modelgrid_wavs[res_len-2])
			model_lhs[res_len] = self.chosen_modelgrid_wavs[res_len-1] + (self.chosen_modelgrid_wavs[res_len-1]-self.chosen_modelgrid_wavs[res_len-2])/2
			model_lhs[1:res_len] = (self.chosen_modelgrid_wavs[1:] + self.chosen_modelgrid_wavs[:res_len-1])/2
			model_widths[:res_len-1] = model_lhs[1:res_len]-model_lhs[:res_len-1]

			# wav_widths: An array containing the relative widths*wavelengths for each point in the spectrum when integrating for photometry
			self.wav_widths = model_widths*self.chosen_modelgrid_wavs

			# get_phot_wavs generates the central wavelengths for each photometric band and populates phot_wavs with them
			self.get_phot_wavs()

		# This is relocated stuff about cloudy line dust attenuation etc. It 
		# If it has not already been populated, populate cloudylinewavs with the wavelengths at which the lines should be inserted
		if "nebular" in list(self.model_comp):
			self.cloudylinewavs = np.loadtxt(models.install_dir + "/tables/nebular/cloudy_linewavs.txt")

			if "dust" in list(self.model_comp):
				self.k_lambda_lines[self.model_comp["dust"]["type"]] = self.get_dust_model(self.model_comp["dust"]["type"], wavs=self.cloudylinewavs)

			self.D_IGM_grid_lines = np.zeros((IGM_redshifts.shape[0], self.cloudylinewavs.shape[0]))
			
			for i in range(IGM_redshifts.shape[0]):
				self.D_IGM_grid_lines[i,:] = interp(self.cloudylinewavs, IGM_wavs_global, D_IGM_grid_global[i,:], left=0., right=1.)

		self.update(self.model_comp)


	""" Loads filter files for the specified filtlist and calculates effective wavelength values which are added to self.photometry """
	def get_phot_wavs(self):

		self.phot_wavs = np.zeros(len(self.filterlist))
		for i in range(len(self.filterlist)):
			filt = np.loadtxt(models.working_dir + "/pipes/filters/" + self.filterlist[i])
			dlambda = models.make_bins(filt[:,0])[1]
			self.phot_wavs[i] = np.round(np.sqrt(np.sum(dlambda*filt[:,1])/np.sum(dlambda*filt[:,1]/filt[:,0]/filt[:,0])), 1)



	""" returns k_lambda values for a given dust curve normalised to A_V = 1. """
	def get_dust_model(self, dust_type, wavs=None):

		if wavs is None:
			wavs = self.chosen_modelgrid_wavs

		if dust_type == "Calzetti":
			dust_corr_calz = np.loadtxt(models.install_dir + "/tables/dust/Calzetti2000_pow_0.77_extrap.txt")
			return interp(wavs, dust_corr_calz[:,0], dust_corr_calz[:,1], right=0)

		elif dust_type == "Cardelli":
			dust_corr_card = np.loadtxt(models.install_dir + "/tables/dust/Cardelli_1989_MW.txt")
			return interp(wavs, dust_corr_card[:,0], dust_corr_card[:,1], right=0)

		elif dust_type == "CF00":
			return wavs/5500.

		else:
			sys.exit("BAGPIPES: Dust type not recognised.")



	""" Loads model grids at specified metallicity, compresses ages and resamples to chosen_modelgrid_wavs. """
	def load_stellar_grid(self, zmet_ind):

		grid_compressed = models.load_stellar_grid(zmet_ind)

		# Resample model grid to the wavelength basis specified in chosen_modelgrid_wavs
		grid_compressed_resampled = np.zeros((grid_compressed.shape[0], self.chosen_modelgrid_wavs.shape[0]))

		for i in xrange(grid_compressed_resampled.shape[0]):
			grid_compressed_resampled[i,:] = interp(self.chosen_modelgrid_wavs, self.original_modelgrid_wavs, grid_compressed[i,:], left=0, right=0)

		# Add the resampled grid to the modelgrids dictionary
		self.modelgrids[str(zmet_ind)] = grid_compressed_resampled



	""" Loads Cloudy nebular continuum and emission lines at specified metallicity and logU. """
	def load_cloudy_grid(self, zmet_ind, logU):

		cloudy_cont_grid, cloudy_line_grid = models.load_cloudy_grid(zmet_ind, logU)

		# Resample the nebular continuum onto the same wavelength basis as the stellar grids have been resampled to
		cloudy_cont_grid_resampled = np.zeros((cloudy_cont_grid.shape[0], self.chosen_modelgrid_wavs.shape[0]))

		for i in range(cloudy_cont_grid_resampled.shape[0]):
			cloudy_cont_grid_resampled[i,:] = interp(self.chosen_modelgrid_wavs, self.original_modelgrid_wavs, cloudy_cont_grid[i,:], left=0, right=0)

		# Add the nebular lines to the resampled nebular continuum
		for i in range(self.cloudylinewavs.shape[0]):
			line_index = np.abs(self.chosen_modelgrid_wavs - self.cloudylinewavs[i]).argmin()
			if line_index != 0 and line_index != self.chosen_modelgrid_wavs.shape[0]-1:
				cloudy_cont_grid_resampled[:,line_index] += cloudy_line_grid[:,i]/((self.chosen_modelgrid_wavs[line_index+1] - self.chosen_modelgrid_wavs[line_index-1])/2.)
		
		# Add the finished cloudy grid to the cloudygrids dictionary
		self.cloudygrids[str(zmet_ind) + str(logU)] = cloudy_cont_grid_resampled



	def update(self, model_components):
		""" 
		Updates the model with new attributes passed in the model_components dictionary. 
		"""

		self.model_comp = model_components

		# sfh: A star formation history object generated using model_components
		self.sfh = star_formation_history.Star_Formation_History(self.model_comp)

		self.living_stellar_mass = {}
		self.living_stellar_mass["total"] = 0.

		# composite_spectrum: will contain the spectrum obtained from adding contributions from all of the SFH components together
		composite_spectrum = np.zeros(self.chosen_modelgrid_wavs.shape[0])

		# composite_lines: will contain the line strengths obtained from adding contributions from all SFH components together
		composite_lines = None

		# total_dust_flux: saves the total flux absorbed by dust to set the normalisation of the dust emission component (erg/s)
		total_dust_flux = 0.

		# Figure out which are the nebular grids in logU closest to the chosen logU and what fraction of each grid should be taken
		if "nebular" in list(self.model_comp):
			high_logU_val = np.min(models.logU_grid[models.logU_grid >= self.model_comp["nebular"]["logU"]])
			low_logU_val =  np.max(models.logU_grid[models.logU_grid < self.model_comp["nebular"]["logU"]])
			high_logU_factor = (self.model_comp["nebular"]["logU"] - low_logU_val)/(high_logU_val - low_logU_val)
			low_logU_factor = 1. - high_logU_factor


		# If dust absorption is requested, make sure the k_lambda values for the requested dust model have been loaded
		if "dust" in list(self.model_comp):
			# Load the k_lambda values for the requested dust model if not already done
			if self.model_comp["dust"]["type"] not in list(self.k_lambda):
				self.k_lambda[self.model_comp["dust"]["type"]] = self.get_dust_model(self.model_comp["dust"]["type"])

			# The CF00 dust model has an extra free parameter n which raises k_lambda to a power, this power is set to 1 for all other dust types
			if self.model_comp["dust"]["type"] == "CF00":
				n = -self.model_comp["dust"]["n"]

			else:
				n = 1.


		# Loop over all star formation history components of the model
		for comp in self.sfh_components:

			# sfr_dt: Array containing the multiplicative factors to be applied to each SSP in the grid to build the CSP.
			sfr_dt = self.sfh.weight_widths[comp]

			# Figure out which are the stellar grids in Zmet closest to the chosen Zmet and what fraction of each grid should be taken, for normal zmet weighting of stellar population and for nebular prescription
			high_zmet_ind = self.zmet_vals[self.zmet_vals < self.model_comp[comp]["metallicity"]].shape[0]

			if high_zmet_ind == self.zmet_vals.shape[0]:
				high_zmet_ind, low_zmet_ind, high_zmet_factor, low_zmet_factor = self.zmet_vals.shape[0]-1, self.zmet_vals.shape[0]-2, 1., 0.

			elif high_zmet_ind == 0:
				low_zmet_ind, high_zmet_factor, low_zmet_factor = 1, 1., 0.

			else:
				low_zmet_ind = high_zmet_ind - 1
				high_zmet_factor = (self.model_comp[comp]["metallicity"] - self.zmet_vals[low_zmet_ind])/(self.zmet_vals[high_zmet_ind] - self.zmet_vals[low_zmet_ind])
				low_zmet_factor = 1 - high_zmet_factor

			# Calculate metallicity contributions
			zmet_weights = np.zeros(self.zmet_vals.shape[0])

			if "metallicity dist" in list(self.model_comp[comp]) and self.model_comp[comp]["metallicity dist"]:
				zmet_factors_highres = (1./self.model_comp[comp]["metallicity"])*np.exp(-self.zmet_vals_highres/self.model_comp[comp]["metallicity"])

				for i in range(zmet_weights.shape[0]):
					zmet_weights[i] = np.sum(0.01*zmet_factors_highres[(self.zmet_vals_highres > self.zmet_lims[i]) & (self.zmet_vals_highres < self.zmet_lims[i+1])])

			else:
				zmet_weights[high_zmet_ind] = high_zmet_factor
				zmet_weights[low_zmet_ind] = low_zmet_factor

			# Check the correct model grids have been loaded
			for i in range(zmet_weights.shape[0]):
				if str(i) not in list(self.modelgrids) and zmet_weights[i] != 0.:
					self.load_stellar_grid(i)

			#Calculate living stellar mass contribution from this SFH component
			comp_living_mass = np.sum(sfr_dt*np.sum(np.expand_dims(zmet_weights, 1)*models.chosen_live_mstar_frac, axis=0))
			self.living_stellar_mass["total"] += comp_living_mass
			self.living_stellar_mass[comp] = np.copy(comp_living_mass)

			interpolated_stellar_grid = np.zeros((self.modelgrids[str(high_zmet_ind)].shape[0], self.modelgrids[str(high_zmet_ind)].shape[1]))

			for i in range(zmet_weights.shape[0]):
				if zmet_weights[i] != 0.:
					interpolated_stellar_grid += zmet_weights[i]*self.modelgrids[str(i)]

			# Calculate how many lines of the grids are affected by birth clouds and what fraction of the final line is affected
			# Otherwise check nothing which required t_bc to be specified has been specified and crash the code if so
			if "t_bc" in list(self.model_comp):
				nlines = models.chosen_age_lhs[models.chosen_age_lhs < self.model_comp["t_bc"]*10**9].shape[0]
				frac_bc = (self.model_comp["t_bc"]*10**9 - models.chosen_age_lhs[nlines-1])/models.chosen_age_widths[nlines-1]

			else:
				if "nebular" in list(self.model_comp) or "dust" in list(self.model_comp) and "eta" in list(self.model_comp["dust"]):
					sys.exit("Bagpipes: t_bc must be specified if nebular emission or more dust for young ages are specified.")


			# This section loads the nebular grids and adds them to the stellar grid if nebular emission has been requested
			if "nebular" in list(self.model_comp):

				# First load any necessary nebular/cloudy models which have not yet been loaded
				for zmet_ind in (high_zmet_ind, low_zmet_ind):
					for logU_val in (high_logU_val, low_logU_val):
						if str(zmet_ind) + str(logU_val) not in list(self.cloudygrids):
							self.load_cloudy_grid(zmet_ind, logU_val)

				# Interpolate the nebular/cloudy grids in Zmet and logU
				interpolated_cloudy_grid = high_zmet_factor*(high_logU_factor*self.cloudygrids[str(high_zmet_ind) + str(high_logU_val)] + low_logU_factor*self.cloudygrids[str(high_zmet_ind) + str(low_logU_val)]) + low_zmet_factor*(high_logU_factor*self.cloudygrids[str(low_zmet_ind) + str(high_logU_val)] + low_logU_factor*self.cloudygrids[str(low_zmet_ind) + str(low_logU_val)])
				interpolated_cloudy_line_grid = high_zmet_factor*(high_logU_factor*models.allcloudylinegrids[str(high_zmet_ind) + str(high_logU_val)] + low_logU_factor*models.allcloudylinegrids[str(high_zmet_ind) + str(low_logU_val)]) + low_zmet_factor*(high_logU_factor*models.allcloudylinegrids[str(low_zmet_ind) + str(high_logU_val)] + low_logU_factor*models.allcloudylinegrids[str(low_zmet_ind) + str(low_logU_val)])

				# Add nebular grid to stellar grid
				interpolated_stellar_grid[:nlines-1, :] += interpolated_cloudy_grid[:nlines-1, :]
				interpolated_stellar_grid[nlines-1, :] += interpolated_cloudy_grid[nlines-1, :]*frac_bc

				interpolated_cloudy_line_grid[nlines-1, :] *= frac_bc

				interpolated_cloudy_lines = np.sum(np.expand_dims(sfr_dt[:nlines], axis=1)*interpolated_cloudy_line_grid[:nlines, :], axis=0)


			# If extra dust on the youngest stellar component has been requested, this section adds it in before the grid is converted from SSPs into a CSP
			if "dust" in list(self.model_comp) and "eta" in list(self.model_comp["dust"]):

				T_dust = 10**(-(self.model_comp["dust"]["eta"] - 1)*self.model_comp["dust"]["Av"]*self.k_lambda[self.model_comp["dust"]["type"]]**n/2.5)

				stellar_grid_no_dust = np.copy(interpolated_stellar_grid[:nlines,:])
				interpolated_stellar_grid[:nlines-1, :] *= T_dust
				interpolated_stellar_grid[nlines-1, :] *= T_dust*frac_bc + (1. - frac_bc)

				total_dust_flux += np.trapz(np.sum(stellar_grid_no_dust - interpolated_stellar_grid[:nlines,:], axis=0), x=self.chosen_modelgrid_wavs)

				if "nebular" in list(self.model_comp):
					interpolated_cloudy_lines *= 10**(-(self.model_comp["dust"]["eta"] - 1)*self.model_comp["dust"]["Av"]*self.k_lambda_lines[self.model_comp["dust"]["type"]]**n/2.5)

			# Obtain CSP by performing a weighed sum over the SSPs and add it to the composite spectrum
			composite_spectrum += np.sum(np.expand_dims(sfr_dt, axis=1)*interpolated_stellar_grid, axis=0)

			if "nebular" in list(self.model_comp):
				if composite_lines is None:
					composite_lines = np.zeros(self.cloudylinewavs.shape[0])
				
				composite_lines += interpolated_cloudy_lines

			# The Hydrogen ionizing continuum is removed here by default
			if self.keep_ionizing_continuum is not True:
				composite_spectrum[self.chosen_modelgrid_wavs < 911.8] = 0.

				if "nebular" in list(self.model_comp):
					composite_lines[self.cloudylinewavs < 911.8] = 0.

		# Apply diffuse dust absorption correction to the composite spectrum
		if "dust" in list(self.model_comp):

			composite_spectrum_nodust = np.copy(composite_spectrum)
			composite_spectrum *= 10**(-self.model_comp["dust"]["Av"]*self.k_lambda[self.model_comp["dust"]["type"]]**n/2.5)
			total_dust_flux += np.trapz(composite_spectrum_nodust - composite_spectrum, x = self.chosen_modelgrid_wavs)

			if "nebular" in list(self.model_comp):
				composite_lines *= 10**(-self.model_comp["dust"]["Av"]*self.k_lambda_lines[self.model_comp["dust"]["type"]]**n/2.5)


		# Apply intergalactic medium absorption to the model spectrum
		if self.model_comp["redshift"] > 0.:
			zred_ind = IGM_redshifts[IGM_redshifts < self.model_comp["redshift"]].shape[0]

			high_zred_factor = (IGM_redshifts[zred_ind] - self.model_comp["redshift"])/(IGM_redshifts[zred_ind] - IGM_redshifts[zred_ind-1])
			low_zred_factor = 1. - high_zred_factor

			# Interpolate IGM transmission from pre-loaded grid
			IGM_trans = self.D_IGM_grid[zred_ind-1,:]*low_zred_factor + self.D_IGM_grid[zred_ind,:]*high_zred_factor
			
			composite_spectrum[(self.chosen_modelgrid_wavs < 1220.) & (self.chosen_modelgrid_wavs > 911.8)] *= IGM_trans

			if "nebular" in list(self.model_comp):
				IGM_trans_lines = self.D_IGM_grid_lines[zred_ind-1,:]*low_zred_factor + self.D_IGM_grid_lines[zred_ind,:]*high_zred_factor

				composite_lines *= IGM_trans_lines
		

		# Add dust emission model if dust keyword "temp" is specified
		if "dust" in list(self.model_comp) and "temp" in list(self.model_comp["dust"]):

			wavs = self.chosen_modelgrid_wavs*10**-10 # Convert wavelength values from microns to metres

			if "beta" in list(self.model_comp["dust"]):
				beta = self.model_comp["dust"]["beta"]

			else:
				beta = 1.5

			# At certain points in parameter space, this factor gets too big to take the exponential of, resulting in math errors
			greybody_factor = self.hc_k/self.model_comp["dust"]["temp"]/wavs

			S_jy_greybody = np.zeros(wavs.shape[0])

			# Therefore only actually evaluate the exponential if greybody_factor is greater than 300, otherwise greybody flux is zero
			S_jy_greybody[greybody_factor < 300.] = (((3*10**8)/wavs[greybody_factor < 300.])**(3+beta))/(np.exp(greybody_factor[greybody_factor < 300.]) - 1.)

			dust_emission = S_jy_greybody*wavs**2 #convert to f_lambda
			dust_emission /= np.trapz(dust_emission, x=self.chosen_modelgrid_wavs)
			dust_emission *= total_dust_flux

			composite_spectrum += dust_emission
			
		# Redshift the model spectrum and put it into erg/s/cm2/A unless z = 0, in which case final units are erg/s/A
		if self.model_comp["redshift"] != 0.:
			composite_spectrum /= (4*np.pi*(interp(self.model_comp["redshift"], z_array, ldist_at_z, left=0, right=0)*3.086*10**24)**2) #convert to observed flux at given redshift in L_sol/s/A/cm^2
			composite_spectrum /= (1+self.model_comp["redshift"]) #reduce flux by a factor of 1/(1+z) to account for redshifting

			if "nebular" in list(self.model_comp):
				composite_lines /= (4*np.pi*(interp(self.model_comp["redshift"], z_array, ldist_at_z, left=0, right=0)*3.086*10**24)**2) #convert to observed flux at given redshift in L_sol/s/A/cm^2

		composite_spectrum *= 3.826*10**33 #convert to erg/s/A/cm^2, or erg/s/A if redshift = 0.

		if "nebular" in list(self.model_comp):
			composite_lines *= 3.826*10**33 #convert to erg/s/cm^2, or erg/s if redshift = 0.

		# spectrum_full: a copy of the full spectrum to be used in plots
		self.spectrum_full = composite_spectrum

		if self.filtlist is not None:
			# Resample the filter profiles onto the redshifted model wavelength grid
			for i in range(len(self.filterlist)):
				self.filt_array[:,i] = interp(self.chosen_modelgrid_wavs*(1.+self.model_comp["redshift"]), self.chosen_modelgrid_wavs, self.filt_array_restframe[:,i], left=0, right=0)

			# photometry: The flux values for the model spectrum observed through the filters specified in filtlist
			self.photometry = np.squeeze(np.sum(np.expand_dims(composite_spectrum*self.wav_widths, axis=1)*self.filt_array, axis=0)/np.sum(self.filt_array*np.expand_dims(self.wav_widths, axis=1), axis=0))
			""" Output photometry array, contains a column of flux values, by default in erg/s/cm^2/A or erg/s/A at redshift zero. """

		# Generate the output spectrum if requested
		if self.output_specwavs is not None:

			# Apply velocity dispersion to the output spectrum if requested
			if "veldisp" in list(self.model_comp):

				specslice = composite_spectrum[(self.chosen_modelgrid_wavs > self.output_specwavs[0]/(1+self.model_comp["redshift"])) & (self.chosen_modelgrid_wavs < self.output_specwavs[-1]/(1+self.model_comp["redshift"]))]

				vres = 3*10**5/np.max(self.R)/2.
				sigma_pix = self.model_comp["veldisp"]/vres
				x_kernel_pix = np.arange(-4*int(sigma_pix+1), 4*int(sigma_pix+1)+1, 1.)
				kernel = (1./np.sqrt(2*np.pi)/sigma_pix)*np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
				composite_spectrum_veldisp = np.convolve(composite_spectrum, kernel, mode="valid")

				# Resample onto the requested wavelength grid
				if x_kernel_pix.shape[0] > 1:
					self.spectrum = np.array([self.output_specwavs, interp(self.output_specwavs, self.chosen_modelgrid_wavs[(x_kernel_pix.shape[0]-1)/2:-(x_kernel_pix.shape[0]-1)/2]*(1+self.model_comp["redshift"]), composite_spectrum_veldisp, left=0, right=0)]).T
					""" Output spectrum array, contains a column of wavelengths in Angstroms and a column of flux values, by default in erg/s/cm^2/A or erg/s/A at redshift zero. """

				else:
					self.spectrum = np.array([self.output_specwavs, interp(self.output_specwavs, self.chosen_modelgrid_wavs*(1+self.model_comp["redshift"]), composite_spectrum_veldisp, left=0, right=0)]).T

			# Otherwise just resample onto the requested wavelength grid
			else:
				self.spectrum = np.array([self.output_specwavs, interp(self.output_specwavs, self.chosen_modelgrid_wavs*(1+self.model_comp["redshift"]), composite_spectrum, left=0, right=0)]).T

		# Apply spectral polynomial if requested
		if "polynomial" in list(self.model_comp):

			polycoefs = []

			while str(len(polycoefs)) in list(self.model_comp["polynomial"]):
				polycoefs.append(self.model_comp["polynomial"][str(len(polycoefs))])

			points = (self.output_specwavs - self.output_specwavs[0])/(self.output_specwavs[-1] - self.output_specwavs[0])

			self.polynomial = cheb(points, polycoefs)

			self.spectrum[:,1] *= self.polynomial

		if "nebular" in list(self.model_comp):
			self.line_fluxes = dict(zip(self.cloudylinelabels, composite_lines))
			""" Dictionary of output emission line fluxes in erg/s/cm^2 or erg/s at redshift zero. """

		# Return output spectrum and photometry in microjanskys instead of erg/s/cm^2/A if requested.
		if self.out_units_spec == "mujy" and self.output_specwavs is not None:
			self.spectrum[:,1] /= (10**-29)*(2.9979*10**18/self.spectrum[:,0]/self.spectrum[:,0])

		if self.out_units_phot == "mujy" and self.filtlist is not None:
			self.photometry /= (10**-29)*(2.9979*10**18/self.phot_wavs/self.phot_wavs)


	def get_restframe_UVJ(self):

		if self.UVJ_filterlist is None:

			# filterlist: a list of the filter file names for UVJ.
			self.UVJ_filterlist = np.loadtxt(models.install_dir + "/filters/UVJ.filtlist", dtype="str")

			# filt_array_restframe: An array to contain the rest frame sampled filter profiles used to generate the model photometry
			self.UVJ_filt_array = np.zeros((self.chosen_modelgrid_wavs.shape[0], 3))

			for i in range(len(self.UVJ_filterlist)):
				filter_raw = np.loadtxt(models.install_dir + "/filters/" + self.UVJ_filterlist[i], usecols=(0, 1))

				self.UVJ_filt_array[:,i] = np.interp(self.chosen_modelgrid_wavs, filter_raw[:,0], filter_raw[:,1], left=0, right=0)
			
			ABzpt_spec = np.array([self.chosen_modelgrid_wavs, (3.*10**18*3631.*10**-23)/(self.chosen_modelgrid_wavs**2)]).T

			self.ABzpt = np.sum(np.expand_dims(ABzpt_spec[:,1]*self.wav_widths, axis=1)*self.UVJ_filt_array, axis=0)/np.sum(np.expand_dims(self.wav_widths, axis=1)*self.UVJ_filt_array, axis=0)

		UVJ = (np.squeeze(np.sum(np.expand_dims(self.spectrum_full*self.wav_widths, axis=1)*self.UVJ_filt_array, axis=0)/np.sum(self.UVJ_filt_array*np.expand_dims(self.wav_widths, axis=1), axis=0)))

		UVJ = -2.5*np.log10((np.squeeze(np.sum(np.expand_dims(self.spectrum_full*self.wav_widths, axis=1)*self.UVJ_filt_array, axis=0)/np.sum(self.UVJ_filt_array*np.expand_dims(self.wav_widths, axis=1), axis=0)))/self.ABzpt)

		return UVJ


	def plot(self, fancy=False, fig=None):
		""" Creates a plot of the model attributes which were requested by the user. """

		if fancy==True:
			self.spectrum[:,1] *= 10**-40
			self.spectrum_full *= 10**-40
			self.photometry *= 10**-40

		#fancy = False
		naxes = 1

		if self.filtlist is not None and self.output_specwavs is not None:
			naxes = 2

		fig, axes = plt.subplots(naxes, figsize=(12, 4.*naxes))

		if naxes == 1:
			axes = [axes]

		ax1 = axes[0]
		ax2 = axes[-1]

		ax2.set_xlabel("$\mathrm{Wavelength\ /\ \AA}$", size=18)

		plt.subplots_adjust(hspace=0.1)

		if self.model_comp["redshift"] != 0:
			if naxes == 2:
				fig.text(0.06, 0.58, "$\mathrm{f_{\lambda}}\ \mathrm{/\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$", size=18, rotation=90)

			else:
				ax1.set_ylabel("$\mathrm{f_{\lambda}}\ \mathrm{/\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$", size=18)

		else:
			if naxes == 2:
				fig.text(0.06, 0.58, "$\mathrm{L_{\lambda}}\ \mathrm{/\ 10^{40}\ erg\ s^{-1}\ \AA^{-1}}$", size=18, rotation=90)

			else:
				ax1.set_ylabel("$\mathrm{f_{\lambda}}\ \mathrm{/\ 10^{40}\ erg\ s^{-1}\ \AA^{-1}}$", size=18)


		if self.filtlist is not None:
			ax2.scatter(self.phot_wavs, self.photometry, color="darkorange", zorder=3, s=150)
			ax2.plot(self.chosen_modelgrid_wavs*(1.+self.model_comp["redshift"]), self.spectrum_full, color="navajowhite", zorder=1)
			ax2.set_xlim(10**(np.log10(self.phot_wavs[0])-0.025), 10**(np.log10(self.phot_wavs[-1])+0.025))
			ax2.set_ylim(0, 1.05*np.max(self.spectrum_full[(self.chosen_modelgrid_wavs*(1+self.model_comp["redshift"]) > ax2.get_xlim()[0]) & (self.chosen_modelgrid_wavs*(1+self.model_comp["redshift"]) < ax2.get_xlim()[1])]))
			ax2.set_xscale("log")
			ax2.set_xticks([5000., 10000., 20000., 50000.,])
			ax2.get_xaxis().set_tick_params(which='minor', size=0)
			ax2.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())


		if self.output_specwavs is not None:
			ax1.plot(self.spectrum[:,0], self.spectrum[:,1], color="sandybrown", zorder=1)
			ax1.set_xlim(self.spectrum[0,0], self.spectrum[-1,0])
			#ax1.set_yticks([0.5, 1.0, 1.5])

		if fancy == True:
			sfh_ax = fig.add_axes([0.54, 0.29, 0.35, 0.17], zorder=10)

			sfh_x = np.zeros(2*self.sfh.ages.shape[0])
			sfh_y = np.zeros(2*self.sfh.sfr.shape[0])

			for j in range(self.sfh.sfr.shape[0]):

				sfh_x[2*j] = self.sfh.age_lhs[j]
				if j+1 < self.sfh.sfr.shape[0]:
					sfh_x[2*j + 1] = self.sfh.age_lhs[j+1]

				sfh_y[2*j] = self.sfh.sfr[j]
				sfh_y[2*j + 1] = self.sfh.sfr[j]

			sfh_x[-2:] = 1.5*10**10

			sfh_ax.plot(cosmo.age(self.model_comp["redshift"]).value - sfh_x*10**-9, sfh_y, color="black")
			sfh_ax.set_xlim(cosmo.age(self.model_comp["redshift"]).value, 0.)
			sfh_ax.set_ylim(0., 1.1*np.max(sfh_y))
			sfh_ax.set_xlabel("$\mathrm{Age\ of\ Universe\ /\ Gyr}$")
			sfh_ax.set_ylabel("$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")

			allspec_ax = fig.add_axes([0.54, 0.57, 0.35, 0.17], zorder=10)
			allspec_ax.plot(self.chosen_modelgrid_wavs*(1.+self.model_comp["redshift"]), self.chosen_modelgrid_wavs*self.spectrum_full, color="black", zorder=1)
			allspec_ax.set_xlim(700., 10**7)
			allspec_ax.set_ylim(5., 2.*np.max(self.chosen_modelgrid_wavs*self.spectrum_full))
			allspec_ax.set_xscale("log")
			allspec_ax.set_yscale("log")
			allspec_ax.set_ylabel("$\mathrm{\lambda L_{\lambda}}\ \mathrm{/\ 10^{40}\ erg\ s^{-1}}$")
			fig.text(0.54+0.135, 0.53, "$\mathrm{Wavelength\ /\ \AA}$")

			allspec_ax.get_xaxis().set_tick_params(which='minor', size=0)
			allspec_ax.get_yaxis().set_tick_params(which='minor', size=0)

		plt.show()
		plt.close(fig)




from __future__ import print_function, division, absolute_import

import numpy as np 
from astropy.io import fits
import os

install_dir = os.path.dirname(os.path.realpath(__file__)) + "/.."
working_dir = os.getcwd()

model_dirs = {}

model_dirs["bc03_miles"] = install_dir + "/tables/stellar/bc03_miles/"
#model_dirs["bc03_basel"] = install_dir + "/tables/stellar/bc03_basel/"
#model_dirs["bpass_bin_100"] = install_dir + "/tables/stellar/bpass_bin_100/"
#model_dirs["bpass_bin_300"] = install_dir + "/tables/stellar/bpass_bin_300/"
#model_dirs["mar05"] = install_dir + "/tables/stellar/mar05/"


def bc03_miles():
	""" Function for loading up the BC03 MILES model library from compressed fits versions. """
	zmet_vals = np.array([0.005, 0.02, 0.2, 0.4, 1., 2.5, 5.])
	len_wavs =  13216
	modelwavs = fits.open(install_dir + "/tables/stellar/bc03_miles/bc03_miles_stellar_grids.fits")[-1].data
	ages = fits.open(install_dir + "/tables/stellar/bc03_miles/bc03_miles_stellar_grids.fits")[-2].data #ages[0] = 0.
	live_mstar_frac = fits.open(install_dir + "/tables/stellar/bc03_miles/bc03_miles_stellar_grids.fits")[-3].data

	return zmet_vals, modelwavs, ages, live_mstar_frac

def bc03_miles_get_grid(zmet_ind):
	return fits.open(install_dir + "/tables/stellar/bc03_miles/bc03_miles_stellar_grids.fits")[zmet_ind+1].data




##### ##### ##### ##### ##### DO NOT DELETE ##### ##### ##### ##### ##### 
# This function loads the uncompressed BC03 model files

"""
def bc03_miles():
	zmet_fnames = np.loadtxt(install_dir + "/tables/stellar/bc03_miles/bc03_miles_fnames.txt", dtype="str")
	zmet_vals = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1])#
	len_wavs =  13216
	modelwavs = np.squeeze(np.genfromtxt(model_dirs["bc03_miles"] + zmet_fnames[0], skip_header = 6, skip_footer=233, usecols=np.arange(1, len_wavs+1), dtype="float"))

	SED_file = open(model_dirs["bc03_miles"] + zmet_fnames[0])
	ages = np.array(SED_file.readline().split(), dtype="float")[1:] #ages[0] = 0.
	SED_file.close()

	mstar_liv = np.loadtxt(install_dir + "/tables/stellar/bc03_miles/live_mass_fractions.txt")

	return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages, mstar_liv

def bc03_miles_get_grid(fname):
	return np.genfromtxt(model_dirs["bc03_miles"] + fname, skip_header = 7, skip_footer=12, usecols=np.arange(1, 13216+1), dtype="float").T
"""



"""
def bc03_basel():
	zmet_fnames = np.loadtxt(install_dir + "/tables/stellar/bc03_basel_fnames.txt", dtype="str")
	zmet_vals = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1])
	len_wavs =  1238
	modelwavs = np.squeeze(np.genfromtxt(model_dirs["bc03_basel"] + zmet_fnames[0], skip_header = 6, skip_footer=233, usecols=np.arange(1, len_wavs+1), dtype="float"))

	SED_file = open(model_dirs["bc03_basel"] + zmet_fnames[0])
	ages = np.array(SED_file.readline().split(), dtype="float")[1:] #ages[0] = 0.
	SED_file.close()

	return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages

def bc03_basel_get_grid(fname):
	len_wavs = 1238
	return np.genfromtxt(model_dirs["bc03_basel"] + fname, skip_header = 7, skip_footer=12, usecols=np.arange(1, len_wavs+1), dtype="float").T



def bpass_bin_100():
	zmet_fnames = np.loadtxt(install_dir + "/tables/stellar/bpass_bin_100_fnames.txt", dtype="str")
	zmet_vals = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.04])
	len_wavs = 100000
	modelwavs = np.squeeze(np.genfromtxt(model_dirs["bpass_bin_100"] + zmet_fnames[0], usecols=(0,), dtype="float"))
	n = np.arange(2, 43)
	ages = 10**(6 + 0.1*(n-2))
	return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages

def bpass_bin_100_get_grid(fname):
	return np.loadtxt(model_dirs["bpass_bin_100"] + fname, dtype="float")[:, 1:]/(10**6)



def bpass_bin_300():
	zmet_fnames = np.loadtxt(install_dir + "/tables/stellar/bpass_bin_300_fnames.txt", dtype="str")
	zmet_vals = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.04])
	len_wavs = 100000
	modelwavs = np.squeeze(np.genfromtxt(model_dirs["bpass_bin_300"] + zmet_fnames[0], usecols=(0,), dtype="float"))
	n = np.arange(2, 43)
	ages = 10**(6 + 0.1*(n-2))
	return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages

def bpass_bin_300_get_grid(fname):
	return np.loadtxt(model_dirs["bpass_bin_300"] + fname, dtype="float")[:, 1:]/(10**6)



def mar05():
	zmet_fnames = np.loadtxt(install_dir + "/tables/stellar/mar05_fnames.txt", dtype="str")
	zmet_vals = np.array([0.001, 0.01, 0.02, 0.04])
	modelwavs = np.loadtxt(model_dirs["mar05"] + zmet_fnames[0], usecols=(2,))[:1221]
	len_wavs = len(modelwavs)
	ages = np.loadtxt(model_dirs["mar05"] + "age_grid.txt")*10**9
	return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages

def mar05_get_grid(fname):
	grid = np.loadtxt(model_dirs["mar05"] + fname, usecols=(3,))
	return grid.reshape(67, 1221).T/(3.826*10**33)
"""



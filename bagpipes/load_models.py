import numpy as np
import setup

model_dirs = {}
model_dirs["bc03_miles"] = setup.install_dir + "/models/bc03_miles/"
#model_dirs["bc03_basel"] = setup.install_dir + "/models/bc03_basel/"
#model_dirs["bpass_bin_100"] = setup.install_dir + "/models/bpass_bin_100/"
#model_dirs["bpass_bin_300"] = setup.install_dir + "/models/bpass_bin_300/"
#model_dirs["mar05"] = setup.install_dir + "/models/mar05/"


def bc03_miles():
    """ Function for loading up the BC03 MILES model library. """
    zmet_fnames = np.loadtxt(setup.install_dir + "/models/bc03_miles_fnames.txt", dtype="str")
    zmet_vals = np.array([0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05, 0.1])
    len_wavs =  13216
    modelwavs = np.squeeze(np.genfromtxt(model_dirs["bc03_miles"] + zmet_fnames[0], skip_header = 6, skip_footer=233, usecols=np.arange(1, len_wavs+1), dtype="float"))

    SED_file = open(model_dirs["bc03_miles"] + zmet_fnames[0])
    ages = np.array(SED_file.readline().split(), dtype="float")[1:] #ages[0] = 0.
    SED_file.close()

    mstar_liv = np.loadtxt(setup.install_dir + "/models/bc03_miles/live_mass_fractions.txt")

    return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages, mstar_liv

def bc03_miles_get_grid(fname):
    return np.genfromtxt(model_dirs["bc03_miles"] + fname, skip_header = 7, skip_footer=12, usecols=np.arange(1, 13216+1), dtype="float").T



"""
def bc03_basel():
    zmet_fnames = np.loadtxt(setup.install_dir + "/models/bc03_basel_fnames.txt", dtype="str")
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
    zmet_fnames = np.loadtxt(setup.install_dir + "/models/bpass_bin_100_fnames.txt", dtype="str")
    zmet_vals = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.04])
    len_wavs = 100000
    modelwavs = np.squeeze(np.genfromtxt(model_dirs["bpass_bin_100"] + zmet_fnames[0], usecols=(0,), dtype="float"))
    n = np.arange(2, 43)
    ages = 10**(6 + 0.1*(n-2))
    return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages

def bpass_bin_100_get_grid(fname):
    return np.loadtxt(model_dirs["bpass_bin_100"] + fname, dtype="float")[:, 1:]/(10**6)




def bpass_bin_300():
    zmet_fnames = np.loadtxt(setup.install_dir + "/models/bpass_bin_300_fnames.txt", dtype="str")
    zmet_vals = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.014, 0.02, 0.03, 0.04])
    len_wavs = 100000
    modelwavs = np.squeeze(np.genfromtxt(model_dirs["bpass_bin_300"] + zmet_fnames[0], usecols=(0,), dtype="float"))
    n = np.arange(2, 43)
    ages = 10**(6 + 0.1*(n-2))
    return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages

def bpass_bin_300_get_grid(fname):
    return np.loadtxt(model_dirs["bpass_bin_300"] + fname, dtype="float")[:, 1:]/(10**6)




def mar05():
    zmet_fnames = np.loadtxt(setup.install_dir + "/models/mar05_fnames.txt", dtype="str")
    zmet_vals = np.array([0.001, 0.01, 0.02, 0.04])
    modelwavs = np.loadtxt(model_dirs["mar05"] + zmet_fnames[0], usecols=(2,))[:1221]
    len_wavs = len(modelwavs)
    ages = np.loadtxt(model_dirs["mar05"] + "age_grid.txt")*10**9
    return zmet_fnames, zmet_vals, len_wavs, modelwavs, ages

def mar05_get_grid(fname):
    grid = np.loadtxt(model_dirs["mar05"] + fname, usecols=(3,))
    return grid.reshape(67, 1221).T/(3.826*10**33)
"""



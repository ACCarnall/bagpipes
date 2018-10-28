from __future__ import print_function, division, absolute_import

import numpy as np
import os
import sys

from astropy.io import fits

from ... import utils
from ..model_galaxy import model_galaxy

if "CLOUDY_DATA_PATH" in list(os.environ):
    cloudy_data_path = os.environ["CLOUDY_DATA_PATH"]

age_lim = 3.*10**7

""" This code is only necessary for making new sets of cloudy nebular
models. It is not called by Bagpipes under normal operation. It is also
currently a little out of date! """


def make_cloudy_sed_file(age, zmet):
    """ Saves a bagpipes spectrum in the correct format to the cloudy
    sed files directory. """

    out_spectrum = get_bagpipes_spectrum(age, zmet, spec_units="mujy")

    out_spectrum[:, 0] = 911.8/out_spectrum[:, 0]

    out_spectrum[:, 1] *= 10**-29  # muJy to erg/s/Hz

    out_spectrum[out_spectrum[:, 1] <= 0., 1] = 9.9*10**-99

    np.savetxt(cloudy_data_path + "/SED/" + utils.model_type + "_age_"
               + "%.5f" % age + "_zmet_" + "%.3f" % zmet + ".sed",
               out_spectrum[::-1, :], header="Energy units: Rydbergs,"
                                             + " Flux units: erg/s/Hz")


def get_bagpipes_spectrum(age, zmet, spec_units="ergscma"):
    """ Makes a bagpipes burst model and returns the spectrum. """
    model_comp = {}

    burst = {}
    burst["age"] = age
    burst["metallicity"] = zmet
    burst["massformed"] = 0.

    model_comp["burst"] = burst
    model_comp["redshift"] = 0.
    model_comp["keep_ionizing"] = True

    model = model_galaxy(model_comp,
                         spec_wavs=utils.gridwavs[utils.model_type],
                         spec_units=spec_units)

    return model.spectrum


def make_cloudy_input_file(age, zmet, logU):
    """ Makes an instructions file for cloudy. """

    # Copy file with emission line names to the correct directory
    if not os.path.exists(cloudy_data_path + "/cloudy_lines.txt"):
        os.system("cp " + utils.install_dir
                  + "/pipes_models/nebular/cloudy_lines.txt "
                  + cloudy_data_path
                  + "/pipes_cloudy_lines.txt")

    logQ = np.log10(4*np.pi*(10.**19)**2*100*2.9979*10**10*10**logU)

    if not os.path.exists(utils.install_dir +
                          "/pipes_models/nebular/cloudy_temp_files"):

        os.mkdir(utils.install_dir + "/pipes_models/nebular/cloudy_temp_files")

    if not os.path.exists(utils.install_dir
                          + "/pipes_models/nebular/cloudy_temp_files/"
                          + utils.model_type):

        os.mkdir(utils.install_dir
                 + "/pipes_models/nebular/cloudy_temp_files/"
                 + utils.model_type)

    path = (utils.install_dir + "/pipes_models/nebular/cloudy_temp_files/"
            + utils.model_type + "/")

    if not os.path.exists(path + "logU_" + "%.1f" % logU
                          + "_zmet_" + "%.3f" % zmet):

        os.mkdir(path + "logU_" + "%.1f" % logU + "_zmet_" + "%.3f" % zmet)

    f = open(path + "logU_" + "%.1f" % logU + "_zmet_" + "%.3f" % zmet
             + "/" + "%.5f" % age + ".in", "w+")

    f.write("########################################\n")

    f.write("##### Input spectrum #####\n")
    f.write("table SED \"" + utils.model_type + "_age_" + "%.5f" % age
            + "_zmet_" + "%.3f" % zmet + ".sed\"\n")

    f.write("########################################\n")

    f.write("##### Geometry and physical conditions #####\n")
    f.write("sphere\n")
    f.write("cosmic rays background\n")
    f.write("hden 2.000 log\n")
    f.write("Q(H) = " + str("%.3f" % logQ) + " log\n")
    f.write("radius 19.000 log\n")
    f.write("abundances old solar 84\n")
    f.write("grains ISM\n")
    f.write("metals grains " + "%.3f" % zmet + "\n")

    # Nitrogen abundances, -0.22 is the depletion factor, the final term
    # is to offset the "metals grains" command.
    if np.log10(zmet) <= -0.63:
        nitrogen_abund = -4.57 + np.log10(zmet) - 0.22 - np.log10(zmet)

    else:
        nitrogen_abund = -3.94 + 2.*np.log10(zmet) - 0.22 - np.log10(zmet)

    elements = ["magnesium", "sulphur", "calcium", "carbon", "oxygen",
                "neon", "silicon", "argon", "iron", "nitrogen", "helium"]

    abundances = np.array([-5.12, -4.79, -8.16, -3.74, -3.29, -3.91, -5.45,
                           -5.44, -6.33, nitrogen_abund,
                           np.log10(0.08096 + 0.02618*zmet)])

    for i in range(len(elements)):
        f.write("element abundance " + elements[i] + " "
                + "%.2f" % abundances[i] + "\n")

    f.write("########################################\n")

    f.write("##### Stopping criteria #####\n")
    f.write("iterate to convergence\n")

    f.write("########################################\n")

    f.write("##### Output continuum and lines #####\n")
    f.write("set save prefix \"" + "%.5f" % age + "\"\n")
    f.write("save last outward continuum \".econ\" units microns\n")
    f.write("save last line list intrinsic absolute column"
            + " \".lines\" \"pipes_cloudy_lines.txt\"\n")

    f.write("########################################")

    f.close()


def run_cloudy_model(age, zmet, logU):
    """ Run an individual cloudy model. """

    make_cloudy_sed_file(age, zmet)
    make_cloudy_input_file(age, zmet, logU)
    os.chdir(utils.install_dir + "/pipes_models/nebular/cloudy_temp_files/"
             + utils.model_type + "/" + "logU_" + "%.1f" % logU
             + "_zmet_" + "%.3f" % zmet)

    os.system(os.environ["CLOUDY_EXE"] + " -r " + "%.5f" % age)
    os.chdir("../../..")


def run_cloudy_grid(nthread, nthreads):

    for i in range(nthread, utils.logU_grid.shape[0], nthreads):
        logU = utils.logU_grid[i]
        for zmet in utils.zmet_vals[utils.model_type]:
            for age in utils.chosen_ages[utils.chosen_ages < age_lim]:
                print("logU: " + str(np.round(logU, 1)) + ", zmet: "
                      + str(np.round(zmet, 4)) + ", age: "
                      + str(np.round(age*10**-9, 5)))

                run_cloudy_model(age*10**-9, zmet, logU)


def extract_cloudy_results(age, zmet, logU, test=False):
    """ Loads the cloudy results from the output files and converts the
    units to L_sol/A for continuum, L_sol for lines. """

    cloudy_lines = np.loadtxt(utils.install_dir
                              + "/pipes_models/nebular/cloudy_temp_files/"
                              + utils.model_type + "/logU_" + "%.1f" % logU
                              + "_zmet_" + "%.3f" % zmet + "/" + "%.5f" % age
                              + ".lines", usecols=(1),
                              delimiter="\t", skiprows=2)

    cloudy_cont = np.loadtxt(utils.install_dir
                             + "/pipes_models/nebular/"
                             + "cloudy_temp_files/" + utils.model_type
                             + "/logU_" + "%.1f" % logU + "_zmet_"
                             + "%.3f" % zmet + "/" + "%.5f" % age + ".econ",
                             usecols=(0, 2))[::-1, :]

    # wavelengths from microns to angstroms
    cloudy_cont[:, 0] *= 10**4

    # continuum from erg/s to erg/s/A.
    cloudy_cont[:, 1] /= cloudy_cont[:, 0]

    # Get bagpipes input spectrum: angstroms, erg/s/A
    input_spectrum = get_bagpipes_spectrum(age, zmet)

    # Total ionizing flux in the bagpipes model in erg/s
    ionizing_spec = input_spectrum[(input_spectrum[:, 0] <= 911.8), 1]
    ionizing_wavs = input_spectrum[(input_spectrum[:, 0] <= 911.8), 0]
    pipes_ionizing_flux = np.trapz(ionizing_spec, x=ionizing_wavs)

    # Total ionizing flux in the cloudy outputs in erg/s
    cloudy_ionizing_flux = np.sum(cloudy_lines) + np.trapz(cloudy_cont[:, 1],
                                                           x=cloudy_cont[:, 0])

    # Normalise cloudy fluxes to the level of the input bagpipes model
    cloudy_lines *= pipes_ionizing_flux/cloudy_ionizing_flux
    cloudy_cont[:, 1] *= pipes_ionizing_flux/cloudy_ionizing_flux

    # Convert cloudy fluxes from erg/s/A to L_sol/A
    cloudy_lines /= 3.826*10**33
    cloudy_cont[:, 1] /= 3.826*10**33

    nlines = utils.gridwavs[utils.model_type].shape[0]
    cloudy_cont_resampled = np.zeros((nlines, 2))

    # Resample the nebular continuum onto wavelengths of stellar models
    cloudy_cont_resampled[:, 0] = utils.gridwavs[utils.model_type]
    cloudy_cont_resampled[:, 1] = np.interp(cloudy_cont_resampled[:, 0],
                                            cloudy_cont[:, 0],
                                            cloudy_cont[:, 1])

    if test:
        old_cont = np.loadtxt("../../pipes_models/" + utils.model_type
                              + "/nebular/zmet_"
                              + "%.1f" % zmet + "_logU_" + "%.1f" % logU
                              + ".neb_cont")[:2, 1:].T

        old_lines = np.loadtxt("../../pipes_models/" + utils.model_type
                               + "/nebular/zmet_"
                               + "%.1f" % zmet + "_logU_" + "%.1f" % logU
                               + ".neb_lines")[1, 1:]

        line_list = np.loadtxt("cloudy_lines.txt", usecols=(0),
                               dtype="str", delimiter="\t")

        important_lines = ["H  1  1215.67A", "O  3  5006.84A",
                           "H  1  6562.81A", "H  1  4861.33A",
                           "N  2  6583.45A"]

        mask = np.isin(line_list, important_lines)

        print("\n")
        print("new cloudy total flux", np.sum(cloudy_lines)
              + np.trapz(cloudy_cont[:, 1], x=cloudy_cont[:, 0]))

        print("old cloudy total flux", np.sum(old_lines)
              + np.trapz(old_cont[:, 1], x=old_cont[:, 0]))

        print("input bagpipes ionizing flux",
              np.trapz(input_spectrum[(input_spectrum[:, 0] <= 911.8), 1],
                       x=input_spectrum[(input_spectrum[:, 0] <= 911.8), 0])
              / (3.826*10**33))

        print("\n")
        print("new and old Lyman alpha fluxes", cloudy_lines[0], old_lines[0])
        print("\n")
        print("new OIII/hbeta",
              np.log10(cloudy_lines[line_list == "O  3  5006.84A"]
                       / cloudy_lines[line_list == "H  1  4861.33A"]),
              "NII/Halpha",
              np.log10(cloudy_lines[line_list == "N  2  6583.45A"]
                       / cloudy_lines[line_list == "H  1  6562.81A"]))

        print("old OIII/hbeta",
              np.log10(old_lines[line_list == "O  3  5006.84A"]
                       / old_lines[line_list == "H  1  4861.33A"]),
              "NII/Halpha",
              np.log10(old_lines[line_lis == "N  2  6583.45A"]
                       / old_lines[line_list == "H  1  6562.81A"]))

        print("\n")

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(old_cont[:, 0], old_cont[:, 1], color="dodgerblue")
        plt.plot(cloudy_cont[:, 0], cloudy_cont[:, 1], color="darkorange")
        plt.xscale("log")
        plt.xlim(800, 10**7)
        plt.show()

        ratio = cloudy_lines/old_lines

        plt.figure()
        plt.axhline(1.)
        plt.scatter(np.arange(124), ratio)
        plt.scatter(np.arange(124)[mask], ratio[mask], color="red")
        plt.show()

    return cloudy_cont_resampled[:, 1], cloudy_lines


def compile_cloudy_grid():

    line_wavs = np.loadtxt(utils.install_dir
                           + "/pipes_models/nebular/cloudy_linewavs.txt")

    for logU in utils.logU_grid:
        for zmet in utils.zmet_vals[utils.model_type]:

            print("logU: " + str(np.round(logU, 1))
                  + ", zmet: " + str(np.round(zmet, 4)))

            mask = (utils.chosen_ages < age_lim)
            contgrid = np.zeros((utils.chosen_ages[mask].shape[0]+1,
                                 utils.gridwavs[utils.model_type].shape[0]+1))

            contgrid[0, 1:] = utils.gridwavs[utils.model_type]
            contgrid[1:, 0] = utils.chosen_ages[utils.chosen_ages < age_lim]

            linegrid = np.zeros((utils.chosen_ages[mask].shape[0]+1,
                                line_wavs.shape[0]+1))

            linegrid[0, 1:] = line_wavs
            linegrid[1:, 0] = utils.chosen_ages[mask]

            for i in range(utils.chosen_ages[mask].shape[0]):
                age = utils.chosen_ages[mask][i]
                cont_fluxes, line_fluxes = extract_cloudy_results(age*10**-9,
                                                                  zmet, logU)

                contgrid[i+1, 1:] = cont_fluxes
                linegrid[i+1, 1:] = line_fluxes

            if not os.path.exists(utils.install_dir
                                  + "/pipes_models/nebular/cloudy_temp_files/"
                                  + utils.model_type + "/grids"):

                os.mkdir(utils.install_dir
                         + "/pipes_models/nebular/cloudy_temp_files/"
                         + utils.model_type + "/grids")

            np.savetxt(utils.install_dir + "/pipes_models/nebular/"
                       "cloudy_temp_files/" + utils.model_type + "/grids/"
                       + "zmet_" + str(zmet) + "_logU_" + str(logU)
                       + ".neb_lines", linegrid)

            np.savetxt(utils.install_dir + "/pipes_models/nebular/"
                       + "cloudy_temp_files/" + utils.model_type + "/grids/"
                       + "zmet_" + str(zmet) + "_logU_" + str(logU)
                       + ".neb_cont", contgrid)

    # Nebular grids
    list_of_hdus_lines = [fits.PrimaryHDU()]
    list_of_hdus_cont = [fits.PrimaryHDU()]

    for logU in utils.logU_grid:
        for zmet in utils.zmet_vals[utils.model_type]:

            line_data = np.loadtxt(utils.install_dir
                                   + "/pipes_models/nebular/cloudy_temp_files/"
                                   + utils.model_type + "/grids/zmet_"
                                   + str(zmet)
                                   + "_logU_" + str(logU) + ".neb_lines")

            hdu_line = fits.ImageHDU(name="zmet_" + "%.3f" % zmet + "_logU_"
                                     + "%.1f" % logU, data=line_data)

            cont_data = np.loadtxt(utils.install_dir
                                   + "/pipes_models/nebular/cloudy_temp_files/"
                                   + utils.model_type + "/grids/zmet_"
                                   + str(zmet)
                                   + "_logU_" + str(logU) + ".neb_cont")

            hdu_cont = fits.ImageHDU(name="zmet_" + "%.3f" % zmet + "_logU_"
                                     + "%.1f" % logU, data=cont_data)

            list_of_hdus_lines.append(hdu_line)
            list_of_hdus_cont.append(hdu_cont)

    hdulist_lines = fits.HDUList(hdus=list_of_hdus_lines)
    hdulist_cont = fits.HDUList(hdus=list_of_hdus_cont)

    os.system("rm " + utils.install_dir
              + "/pipes_models/nebular/cloudy_temp_files/"
              + utils.model_type + "/grids/" + utils.model_type
              + "_nebular_line_grids.fits")

    os.system("rm " + utils.install_dir
              + "/pipes_models/nebular/cloudy_temp_files/"
              + utils.model_type + "/grids/" + utils.model_type
              + "_nebular_cont_grids.fits")

    hdulist_lines.writeto(utils.install_dir
                          + "/pipes_models/nebular/cloudy_temp_files/"
                          + utils.model_type
                          + "/grids/" + utils.model_type
                          + "_nebular_line_grids.fits")

    hdulist_cont.writeto(utils.install_dir
                         + "/pipes_models/nebular/cloudy_temp_files/"
                         + utils.model_type
                         + "/grids/" + utils.model_type
                         + "_nebular_cont_grids.fits")

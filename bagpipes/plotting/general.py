from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from distutils.spawn import find_executable
from scipy.ndimage.filters import gaussian_filter

from .. import utils

tex_on = find_executable("latex")

if not tex_on:
    print("Bagpipes: Latex distribution not found, plots may look strange.")


latex_names = {"redshift": "z",
               "metallicity": "Z",
               "massformed": "\\mathrm{log_{10}(M",
               "mass": "\\mathrm{log_{10}(M_*",
               "stellar_mass": "\\mathrm{log_{10}(M_*",
               "tau": "\\tau",
               "alpha": "\\alpha",
               "beta": "\\beta",
               "age": "\\mathrm{Age}",
               "age_min": "\\mathrm{Min\\ Age}",
               "age_max": "\\mathrm{Max\\ Age}",
               "Av": "{A_V}",
               "n": "n",
               "veldisp": "\\sigma_{vel}",
               "0": "\\mathrm{N}0",
               "1": "\\mathrm{N}1",
               "2": "\\mathrm{N}2",
               "3": "\\mathrm{N}3",
               "4": "\\mathrm{N}4",
               "5": "\\mathrm{N}5",
               "6": "\\mathrm{N}6",
               "7": "\\mathrm{N}7",
               "8": "\\mathrm{N}8",
               "9": "\\mathrm{N}9",
               "10": "\\mathrm{N}10",
               "sfr": "\\mathrm{SFR}",
               "mass_weighted_age": "\\mathrm{Age_{MW}}",
               "tform": "\\mathrm{t_{form}}",
               "tquench": "\\mathrm{t_{quench}}",
               "ssfr": "\\mathrm{log_{10}(sSFR",
               "sig_exp": "\\Delta",
               "prob": "P",
               "mu": "\\mu",
               "sigma": "\\sigma",
               "tau_q": "\\tau_\\mathrm{quench}",
               "length": "l",
               "norm": "n",
               "scaling": "s",
               "t_bc": "t_{BC}"}

latex_units = {"metallicity": "Z_{\\odot}",
               "massformed": "M_{\\odot})}",
               "mass": "M_{\\odot})}",
               "stellar_mass": "M_{\\odot})}",
               "tau": "\\mathrm{Gyr}",
               "age": "\\mathrm{Gyr}",
               "age_min": "\\mathrm{Gyr}",
               "age_max": "\\mathrm{Gyr}",
               "Av": "\\mathrm{mag}",
               "veldisp": "\\mathrm{km s^{-1}}",
               "sfr": "\\mathrm{M_\\odot\\ yr}^{-1}",
               "ssfr": "\\mathrm{yr}^{-1})}",
               "mass_weighted_age": "\\mathrm{Gyr}",
               "tform": "\\mathrm{Gyr}",
               "tau_q": "\\mathrm{Gyr}",
               "tquench": "\\mathrm{Gyr}",
               "t_bc": "\\mathrm{Gyr}"}

latex_comps = {"dblplaw": "dpl",
               "exponential": "exp",
               "constant": "const",
               "delayed": "del",
               "calibration": "calib",
               "nebular": "neb",
               "lognormal": "lnorm"}


def update_rcParams():
    mpl.rcParams["lines.linewidth"] = 2.
    mpl.rcParams["axes.linewidth"] = 1.5
    mpl.rcParams["axes.labelsize"] = 18.
    mpl.rcParams["xtick.top"] = True
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.right"] = True
    mpl.rcParams["ytick.labelsize"] = 14
    mpl.rcParams["ytick.direction"] = "in"

    if tex_on:
        mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        mpl.rc('text', usetex=True)
        mpl.rcParams["text.usetex"] = True

    else:
        mpl.rcParams["text.usetex"] = False


def make_hist_arrays(x, y):
    """ convert x and y arrays for a line plot to a histogram plot. """
    hist_x = np.c_[x[:-1], x[1:]].flatten()
    hist_y = np.c_[y, y].flatten()

    return hist_x, hist_y


def hist1d(samples, ax, smooth=False, label=None, color="orange",
           percentiles=True, zorder=4, bins=50):

    if color == "orange":
        color1 = "darkorange"
        color2 = "navajowhite"
        alpha = 0.7

    if color == "purple":
        color1 = "purple"
        color2 = "purple"
        alpha = 0.4

    if color == "blue":
        color1 = "blue"
        color2 = "dodgerblue"
        alpha = 0.6

    if label is not None:
        x_label = fix_param_names([label])
        ax.set_xlabel(x_label)

    width = samples.max() - np.max([samples.min(), -99.])
    range = (np.max([samples.min(), -99.]) - width/10.,
             samples.max() + width/10.)

    y, x = np.histogram(samples, bins=bins, density=True, range=range)

    y = gaussian_filter(y, 1.5)

    if smooth:
        x_midp = (x[:-1] + x[1:])/2.
        ax.plot(x_midp, y, color=color1, zorder=zorder-1)
        ax.fill_between(x_midp, np.zeros_like(y), y,
                        color=color2, alpha=alpha, zorder=zorder-2)
        ax.plot([x_midp[0], x_midp[0]], [0, y[0]],
                color=color1, zorder=zorder-1)

        ax.plot([x_midp[-1], x_midp[-1]], [0, y[-1]],
                color=color1, zorder=zorder-1)

    else:
        x_hist, y_hist = make_hist_arrays(x, y)
        ax.plot(x_hist, y_hist, color="black")

    if percentiles:
        for percentile in [16, 50, 84]:
            ax.axvline(np.percentile(samples, percentile), linestyle="--",
                       color="black", zorder=zorder, lw=3)

    ax.set_ylim(bottom=0)
    ax.set_xlim(range)
    auto_x_ticks(ax, nticks=3.)
    plt.setp(ax.get_yticklabels(), visible=False)


def auto_x_ticks(ax, nticks=5.):

        spacing = 1./nticks

        width = ax.get_xlim()[1] - ax.get_xlim()[0]
        tick_locs = np.arange(ax.get_xlim()[0] + spacing/2.*width,
                              ax.get_xlim()[1], spacing*width)

        if tick_locs.max() < 0:
            n_decimals = 0

        else:
            n_decimals = -int(np.log10(tick_locs.max()))+1

        for i in range(tick_locs.shape[0]):
            tick_locs[i] = np.round(tick_locs[i], decimals=n_decimals)

        while ((tick_locs[1:] - tick_locs[:-1])/width).min() < (1./(nticks+1)):
            tick_locs = np.arange(ax.get_xlim()[0] + spacing/2.*width,
                                  ax.get_xlim()[1], spacing*width)
            n_decimals += 1
            for i in range(tick_locs.shape[0]):
                tick_locs[i] = np.round(tick_locs[i], decimals=n_decimals)

        ax.set_xticks(tick_locs)


def fix_param_names(fit_params):
    new_params = []

    if not isinstance(fit_params, list):
        fit_params = [fit_params]

    for fit_param in fit_params:
        split = fit_param.split(":")

        if len(split) == 1:
            comp = None
            param = split[0]

        if len(split) == 2:
            comp = split[0]
            param = split[1]

        if param in list(latex_names):
            new_param = latex_names[param]

            if comp is not None:
                if comp in list(latex_comps):
                    new_param += "_\\mathrm{" + latex_comps[comp] + "}"
                else:
                    new_param += "_\\mathrm{" + comp + "}"

            if param in list(latex_units):
                new_param = new_param + "/" + latex_units[param]

            new_param = "$" + new_param + "$"

        else:
            new_param = fit_param

        new_params.append(new_param)

    if len(new_params) == 1:
        new_params = new_params[0]

    return new_params


def auto_axis_label(ax, y_scale, z_non_zero=True, log_x=False):

    if tex_on:
        if z_non_zero:
            ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{"
                          + str(y_scale)
                          + "}\\ erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}}$")

        else:
            ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{"
                          + str(y_scale)
                          + "}\\ erg\\ s^{-1}\\ \\AA^{-1}}$")

        if log_x:
            ax.set_xlabel("$\\mathrm{log_{10}}\\big(\\lambda / \\mathrm{\\AA}"
                          + "\\big)$")

        else:
            ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")

    else:
        if z_non_zero:
            ax.set_ylabel("f_lambda / 10^" + str(y_scale)
                          + " erg s^-1 cm^-2 A^-1")

        else:
            ax.set_ylabel("f_lambda / 10^" + str(y_scale)
                          + " erg s^-1 A^-1")

        if log_x:
            ax.set_xlabel("log_10(lambda / A)")

        else:
            ax.set_xlabel("lambda / A")


def add_z_axis(ax, z_on_y=False, zvals=[0, 0.5, 1, 2, 4, 10]):

    if z_on_y:
        ax2 = ax.twinx()
        ax2.set_yticks(np.interp(zvals, utils.z_array, utils.age_at_z))
        ax2.set_yticklabels(["$" + str(z) + "$" for z in zvals])
        ax2.set_ylim(ax.get_ylim())

    else:
        ax2 = ax.twiny()
        ax2.set_xticks(np.interp(zvals, utils.z_array, utils.age_at_z))
        ax2.set_xticklabels(["$" + str(z) + "$" for z in zvals])
        ax2.set_xlim(ax.get_xlim())

    if tex_on:
        if z_on_y:
            ax2.set_ylabel("$\\mathrm{Redshift}$")

        else:
            ax2.set_xlabel("$\\mathrm{Redshift}$")

    else:
        if z_on_y:
            ax2.set_xlabel("Redshift")

        else:
            ax2.set_xlabel("Redshift")

    return ax2

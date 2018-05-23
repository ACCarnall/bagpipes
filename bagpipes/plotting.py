from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from distutils.spawn import find_executable

from .utils import *


if find_executable("latex"):
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    mpl.rc('text', usetex=True)

    tex_on = True
    mpl.rcParams["text.usetex"] = True


else:
    tex_on = False
    mpl.rcParams["text.usetex"] = False


mpl.rcParams["lines.linewidth"] = 2.
mpl.rcParams["axes.linewidth"] = 1.5
mpl.rcParams["axes.labelsize"] = 18.
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["ytick.direction"] = "in"


def plot_model_galaxy(model, show=True):

    naxes = 1

    if (model.filtlist is not None and model.spec_wavs is not None):
        naxes = 2

    fig = plt.figure(figsize=(12, 4.*naxes))
    gs = mpl.gridspec.GridSpec(naxes, 1, hspace=0.3)

    if model.spec_wavs is not None:
        spec_ax = plt.subplot(gs[0, 0])
        plot_spectrum(model.spectrum, spec_ax)

    if (model.filtlist is not None and model.spec_wavs is not None):
        phot_ax = plt.subplot(gs[1, 0])
        plot_model_photometry(model, phot_ax)

    elif model.filtlist is not None:
        phot_ax = plt.subplot(gs[0, 0])
        plot_model_photometry(model, phot_ax)

    if show:
        plt.show()
        plt.close(fig)

    return fig


def plot_galaxy(galaxy, show=True):

    naxes = 1

    if (galaxy.photometry_exists and galaxy.spectrum_exists):
        naxes = 2

    fig = plt.figure(figsize=(12, 4.*naxes))
    gs = mpl.gridspec.GridSpec(naxes, 1, hspace=0.3)

    if galaxy.spectrum_exists:
        spec_ax = plt.subplot(gs[0, 0])
        plot_spectrum(galaxy.spectrum, spec_ax)

    if galaxy.photometry_exists and galaxy.spectrum_exists:
        phot_ax = plt.subplot(gs[1, 0])
        plot_observed_photometry(galaxy, phot_ax)

    elif galaxy.photometry_exists:
        phot_ax = plt.subplot(gs[0, 0])
        plot_observed_photometry(galaxy, phot_ax)

    if show:
        plt.show()
        plt.close(fig)

    return fig


def plot_spectrum(spectrum, ax, x_ticks=None, zorder=3):

    # Sort out axis limits
    ymax = 1.05*np.max(spectrum[:, 1])

    y_scale = -int(np.log10(ymax))+1

    ax.set_ylim(0., ymax*10**y_scale)
    ax.set_xlim(spectrum[0, 0], spectrum[-1, 0])

    # Plot the data
    if spectrum.shape[1] == 2:
        ax.plot(spectrum[:, 0], spectrum[:, 1]*10**y_scale,
                color="sandybrown", zorder=zorder)

    elif spectrum.shape[1] == 3:
        ax.plot(spectrum[:, 0], spectrum[:, 1]*10**y_scale,
                color="dodgerblue", zorder=zorder, lw=1)

        ax.fill_between(spectrum[:, 0],
                        (spectrum[:, 1] - spectrum[:, 2])*10**y_scale,
                        (spectrum[:, 1] + spectrum[:, 2])*10**y_scale,
                        color="dodgerblue", zorder=zorder-1, alpha=0.75,
                        linewidth=0)

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    # Sort out axis labels
    if tex_on:
        ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{-"
                      + str(y_scale)
                      + "}\\ erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}}$")

        ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")

    else:
        ax.set_ylabel("f_lambda / 10^-" + str(y_scale)
                      + " erg s^-1 cm^-2 A^-1")

        ax.set_xlabel("lambda / A")

    return ax


def plot_model_photometry(model, ax, x_ticks=None, zorder=3):

    # Sort out axis limits
    xmin = np.log10(model.eff_wavs[0])-0.025
    xmax = np.log10(model.eff_wavs[-1])+0.025
    ax.set_xlim(xmin, xmax)

    redshifted_wavs = model.chosen_wavs*(1.+model.model_comp["redshift"])

    spec_mask = ((redshifted_wavs > 10**xmin) & (redshifted_wavs < 10**xmax))

    ymax = 1.05*np.max(model.spectrum_full[spec_mask])

    y_scale = -int(np.log10(ymax))+1

    ax.set_ylim(0., ymax*10**y_scale)

    # Plot the data
    ax.plot(np.log10(redshifted_wavs),
            model.spectrum_full*10**y_scale, color="navajowhite",
            zorder=zorder)

    ax.scatter(np.log10(model.eff_wavs), model.photometry*10**y_scale,
               color="darkorange", s=150, zorder=zorder+1)

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    # Sort out axis labels
    if tex_on:
        ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{-"
                      + str(y_scale)
                      + "}\\ erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}}$")

        ax.set_xlabel("$\\mathrm{log_{10}}"
                      + "\\Big(\\lambda / \\mathrm{\\AA}\\Big)$")

    else:
        ax.set_ylabel("f_lambda / 10^-"
                      + str(y_scale)
                      + " erg s^-1 cm^-2 A^-1")

        ax.set_xlabel("lambda / A")

    return ax


def plot_observed_photometry(galaxy, ax, x_ticks=None, zorder=3):
    # Sort out axis limits
    ax.set_xlim((np.log10(galaxy.eff_wavs[0])-0.025),
                (np.log10(galaxy.eff_wavs[-1])+0.025))

    ymax = 1.05*np.max(galaxy.photometry[:, 1]+galaxy.photometry[:, 2])

    y_scale = -int(np.log10(ymax))+1

    ax.set_ylim(0., ymax*10**y_scale)

    # Plot the data
    ax.errorbar(np.log10(galaxy.photometry[:, 0]),
                galaxy.photometry[:, 1]*10**y_scale,
                yerr=galaxy.photometry[:, 2]*10**y_scale, lw=1.0,
                linestyle=" ", capsize=3, capthick=1, zorder=2, color="black")

    ax.scatter(np.log10(galaxy.photometry[:, 0]),
               galaxy.photometry[:, 1]*10**y_scale, color="blue", s=75,
               zorder=3, linewidth=1, facecolor="blue", edgecolor="black")

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    # Sort out axis labels
    if tex_on:
        ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{-"
                      + str(y_scale)
                      + "}\\ erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}}$")

        ax.set_xlabel("$\\mathrm{log_{10}}\\Big(\\lambda / \\mathrm{\\AA}"
                      + "\\Big)$")

    else:
        ax.set_ylabel("f_lambda / 10^-" + str(y_scale)
                      + " erg s^-1 cm^-2 A^-1")

        ax.set_xlabel("lambda / A")

    return ax


def auto_x_ticks(ax):

        width = ax.get_xlim()[1] - ax.get_xlim()[0]
        tick_locs = np.arange(ax.get_xlim()[0] + 0.1*width, ax.get_xlim()[1],
                              0.2*width)

        for i in range(tick_locs.shape[0]):
            tick_locs[i] = np.round(tick_locs[i],
                                    decimals=-int(np.log10(tick_locs[i]))+1)

        ax.set_xticks(tick_locs)

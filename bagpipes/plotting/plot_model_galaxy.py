from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *
from .plot_spectrum import add_spectrum


def plot_model_galaxy(model, show=True, color="default"):
    """ Make a quick plot of an individual model galaxy. """

    update_rcParams()

    naxes = 1
    if (model.filt_list is not None and model.spec_wavs is not None):
        naxes = 2

    fig = plt.figure(figsize=(12, 4.*naxes))
    gs = mpl.gridspec.GridSpec(naxes, 1, hspace=0.3)

    if model.spec_wavs is not None:
        spec_ax = plt.subplot(gs[0, 0])
        add_spectrum(model.spectrum, spec_ax, color=color,
                     z_non_zero=model.model_comp["redshift"])

        axes = [spec_ax]

    if (model.filt_list is not None and model.spec_wavs is not None):
        phot_ax = plt.subplot(gs[1, 0])
        add_model_photometry(model, phot_ax)
        axes.append(phot_ax)

    elif model.filt_list is not None:
        phot_ax = plt.subplot(gs[0, 0])
        add_model_photometry(model, phot_ax)
        axes = [phot_ax]

    if show:
        plt.show()
        plt.close(fig)

    return fig, axes


def add_model_photometry(model, ax, x_ticks=None, zorder=4, colorscheme=None):
    """ Adds model photometry to the passed axis. """

    color1 = "navajowhite"
    color2 = "darkorange"

    if colorscheme == "bw":
        color1 = "gray"
        color2 = "black"

    # Sort out axis limits
    xmin = np.log10(model.filter_set.eff_wavs.min())-0.025
    xmax = np.log10(model.filter_set.eff_wavs.max())+0.025
    ax.set_xlim(xmin, xmax)

    redshifted_wavs = model.wavelengths*(1.+model.model_comp["redshift"])

    spec_mask = ((redshifted_wavs > 10**xmin) & (redshifted_wavs < 10**xmax))

    ymax = 1.05*np.max(model.spectrum_full[spec_mask])

    y_scale = int(np.log10(ymax))-1

    ax.set_ylim(0., ymax*10**-y_scale)

    # Plot the data
    ax.plot(np.log10(redshifted_wavs),
            model.spectrum_full*10**-y_scale, color=color1,
            zorder=zorder-1)

    ax.scatter(np.log10(model.filter_set.eff_wavs),
               model.photometry*10**-y_scale,
               color=color2, s=150, zorder=zorder)

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    auto_axis_label(ax, y_scale, log_x=True,
                    z_non_zero=model.model_comp["redshift"])

    return ax


def add_full_spectrum(model, ax, color="darkorange", lw=1):
    log_wavs = np.log10(model.wavelengths)
    wav_mask = (log_wavs > 2.75) & (log_wavs < 6.75)

    spec_full = model.spectrum_full*model.lum_flux*model.wavelengths
    spec_full = spec_full[wav_mask]

    ax.plot(log_wavs[wav_mask], np.log10(spec_full), color=color, lw=lw)

    ax.set_xlim(2.75, 6.75)


def plot_full_spectrum(model, show=True):
    """ Make a quick plot of an individual model galaxy. """

    update_rcParams()

    naxes = 1

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot()

    add_full_spectrum(model, ax)

    # Set axis labels
    if tex_on:

        ax.set_ylabel("$\\mathrm{log_{10}}\\big(\\mathrm{\\lambda "
                      + "L_{\\lambda}}\\ \\mathrm{/\\ erg\\ s^{-1}}\\big)$")

        ax.set_xlabel("$\\mathrm{log_{10}}\\big(\\lambda / \\mathrm{\\AA}"
                      + "\\big)$")

    else:
        ax.set_ylabel("f_lambda / erg s^-1")

        ax.set_xlabel("log_10(lambda / A)")

    if show:
        plt.show()
        plt.close(fig)

    else:
        return fig, ax

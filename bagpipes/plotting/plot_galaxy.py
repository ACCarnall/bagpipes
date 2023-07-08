from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *
from .plot_spectrum import add_spectrum


def plot_galaxy(galaxy, show=True, return_y_scale=False, y_scale_spec=None):
    """ Make a quick plot of the data loaded into a galaxy object. """

    update_rcParams()

    naxes = 1
    if (galaxy.photometry_exists and galaxy.spectrum_exists):
        naxes = 2

    y_scale = []

    fig = plt.figure(figsize=(12, 4.*naxes))
    gs = mpl.gridspec.GridSpec(naxes, 1, hspace=0.3)

    # Add observed spectroscopy to plot
    if galaxy.spectrum_exists:
        spec_ax = plt.subplot(gs[0, 0])

        y_scale_spec = add_spectrum(galaxy.spectrum, spec_ax,
                                    y_scale=y_scale_spec)

        if galaxy.photometry_exists:
            add_observed_photometry_linear(galaxy, spec_ax,
                                           y_scale=y_scale_spec)

        axes = [spec_ax]
        y_scale = [y_scale_spec]

    # Add observed photometry to plot
    if galaxy.photometry_exists and galaxy.spectrum_exists:
        phot_ax = plt.subplot(gs[1, 0])
        y_scale_phot = float(add_observed_photometry(galaxy, phot_ax))
        y_scale.append(y_scale_phot)
        axes.append(phot_ax)

    elif galaxy.photometry_exists:
        phot_ax = plt.subplot(gs[0, 0])
        y_scale_phot = float(add_observed_photometry(galaxy, phot_ax))
        y_scale = [y_scale_phot]
        axes = [phot_ax]

    if show:
        plt.show()
        plt.close(fig)

    if return_y_scale:
        return fig, axes, y_scale

    return fig, axes


def add_observed_photometry(galaxy, ax, x_ticks=None, zorder=4, ptsize=40,
                            y_scale=None, lw=1., skip_no_obs=False,
                            label=None, color="blue", marker="o"):
    """ Adds photometric data to the passed axes. """

    photometry = np.copy(galaxy.photometry)

    if skip_no_obs:
        mask = (photometry[:, 1] != 0.)
        photometry = photometry[mask, :]

    # Sort out axis limits
    ax.set_xlim((np.log10(galaxy.filter_set.eff_wavs.min()) - 0.025),
                (np.log10(galaxy.filter_set.eff_wavs.max()) + 0.025))

    mask = (photometry[:, 1] > 0.)
    ymax = 1.1*np.nanmax((photometry[:, 1]+photometry[:, 2])[mask])

    if y_scale is None:
        y_scale = int(np.log10(ymax))-1

    ax.set_ylim(0., ymax*10**-y_scale)

    # Plot the data
    ax.errorbar(np.log10(photometry[:, 0]),
                photometry[:, 1]*10**-y_scale,
                yerr=photometry[:, 2]*10**-y_scale, lw=lw,
                linestyle=" ", capsize=3, capthick=1, zorder=zorder-1,
                color="black")

    ax.scatter(np.log10(photometry[:, 0]),
               photometry[:, 1]*10**-y_scale, color=color, s=ptsize,
               zorder=zorder, linewidth=lw, facecolor=color,
               edgecolor="black", label=label, marker=marker)

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    auto_axis_label(ax, y_scale, log_x=True)

    return y_scale


def add_observed_photometry_linear(galaxy, ax, zorder=4, y_scale=None,
                                   skip_no_obs=False, ptsize=40, lw=1.,
                                   marker="o", label=None, color="blue"):
    """ Adds photometric data to the passed axes without doing any
    manipulation of the axes or labels. """

    photometry = np.copy(galaxy.photometry)

    if skip_no_obs:
        mask = (photometry[:, 1] != 0.)
        photometry = photometry[mask, :]

    mask = (photometry[:, 1] > 0.)
    ymax = 1.05*np.nanmax((photometry[:, 1]+photometry[:, 2])[mask])

    if not y_scale:
        y_scale = int(np.log10(ymax))-1

    # Plot the data
    ax.errorbar(photometry[:, 0], photometry[:, 1]*10**-y_scale,
                yerr=photometry[:, 2]*10**-y_scale, lw=lw,
                linestyle=" ", capsize=3, capthick=lw, zorder=zorder-1,
                color="black")

    ax.scatter(photometry[:, 0], photometry[:, 1]*10**-y_scale, color=color,
               s=ptsize, zorder=zorder, linewidth=lw, facecolor=color,
               edgecolor="black", marker=marker, label=label)

    auto_axis_label(ax, y_scale, log_x=False)

    return ax

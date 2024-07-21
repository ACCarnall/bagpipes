from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *


def plot_sfh(sfh, show=True, save=False):
    """ Make a quick plot of an individual sfh. """

    update_rcParams()

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot()

    add_sfh(sfh, ax)

    if save:
        plt.savefig("model_sfh.pdf", bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax


def add_sfh(sfh, ax, zorder=4, color="black", z_axis=True, lw=2,
            zvals=[0, 0.5, 1, 2, 4, 10], alpha=1, ls="-", label=None):
    """ Creates a plot of sfr(t) for a given star-formation history. """

    # Plot the sfh
    ax.plot((sfh.age_of_universe - sfh.ages)*10**-9, sfh.sfh,
            color=color, zorder=zorder, lw=lw, alpha=alpha, ls=ls, label=label)

    # Set limits
    ax.set_xlim(sfh.age_of_universe*10**-9, 0.)
    ax.set_ylim(bottom=0.)

    # Add redshift axis along the top
    if z_axis:
        z_axis = add_z_axis(ax, zvals=zvals)

    # Add labels
    if tex_on:
        ax.set_ylabel("$\\mathrm{SFR\\ /\\ M_\\odot\\ \\mathrm{yr}^{-1}}$")
        ax.set_xlabel("$\\mathrm{Age\\ of\\ Universe\\ /\\ Gyr}$")

    else:
        ax.set_ylabel("SFR / M_sol yr^-1")
        ax.set_xlabel("Age of Universe / Gyr")

    return z_axis

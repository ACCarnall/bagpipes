from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *

from .. import utils
from .. import config


def plot_sfh_posterior(fit, show=False, save=True, colorscheme="bw"):
    """ Make a plot of the SFH posterior. """

    update_rcParams()

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot()

    add_sfh_posterior(fit, ax, colorscheme=colorscheme)

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_sfh.pdf"
        plt.savefig(plotpath, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax


def add_sfh_posterior(fit, ax, colorscheme="bw", z_axis=True, zorder=4):

    color1 = "black"
    color2 = "gray"
    alpha = 0.6

    if colorscheme == "irnbru":
        color1 = "darkorange"
        color2 = "navajowhite"
        alpha = 0.6

    if colorscheme == "purple":
        color1 = "purple"
        color2 = "purple"
        alpha = 0.4

    # Calculate median redshift and median age of Universe
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])

    else:
        redshift = fit.fitted_model.model_components["redshift"]

    age_of_universe = np.interp(redshift, utils.z_array, utils.age_at_z)

    # Calculate median and confidence interval for SFH posterior
    post = np.percentile(fit.posterior.samples["sfh"], (16, 50, 84), axis=0).T

    # Plot the SFH
    x = age_of_universe - fit.posterior.sfh.ages*10**-9

    ax.plot(x, post[:, 1], color=color1, zorder=zorder+1)
    ax.fill_between(x, post[:, 0], post[:, 2], color=color2,
                    alpha=alpha, zorder=zorder, lw=0)

    ax.set_ylim(0., np.max([ax.get_ylim()[1], 1.1*np.max(post[:, 2])]))
    ax.set_xlim(age_of_universe, 0)

    # Add redshift axis along the top
    if z_axis:
        ax2 = add_z_axis(ax)

    # Set axis labels
    if tex_on:
        ax.set_ylabel("$\\mathrm{SFR\\ /\\ M_\\odot\\ \\mathrm{yr}^{-1}}$")
        ax.set_xlabel("$\\mathrm{Age\\ of\\ Universe\\ /\\ Gyr}$")

    else:
        ax.set_ylabel("SFR / M_sol yr^-1")
        ax.set_xlabel("Age of Universe / Gyr")

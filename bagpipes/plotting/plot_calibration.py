from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *


def plot_calibration(fit, save=True, show=False):
    """ Plot the posterior of the calibration spectral correction. """

    update_rcParams()

    fig = plt.figure()
    ax = plt.subplot()

    ax = add_calibration(fit, ax)

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_cal.pdf"
        plt.savefig(plotpath, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax

def add_calibration(fit, ax):

    fit.posterior.get_advanced_quantities()

    wavs = fit.galaxy.spectrum[:, 0]
    samples = fit.posterior.samples["calib"]
    post = np.percentile(samples, (16, 50, 84), axis=0).T

    ax.plot(wavs, post[:, 0], color="navajowhite", zorder=10)
    ax.plot(wavs, post[:, 1], color="darkorange", zorder=10)
    ax.plot(wavs, post[:, 2], color="navajowhite", zorder=10)
    ax.fill_between(wavs, post[:, 0], post[:, 2], lw=0,
                    color="navajowhite", alpha=0.75, zorder=9)

    ax.set_xlim(wavs[0], wavs[-1])

    if tex_on:
        ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
        ax.set_ylabel("$\\mathrm{Spectrum\\ multiplied\\ by}$")

    else:
        ax.set_xlabel("lambda / A")
        ax.set_ylabel("Spectrum multiplied by")

    return ax

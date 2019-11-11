from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *
from .plot_galaxy import plot_galaxy


def plot_spectrum_posterior(fit, show=False, save=True):
    """ Plot the observational data and posterior from a fit object. """

    fit.posterior.get_advanced_quantities()

    update_rcParams()

    # First plot the observational data
    fig, ax, y_scale = plot_galaxy(fit.galaxy, show=False, return_y_scale=True)

    if fit.galaxy.spectrum_exists:
        add_spectrum_posterior(fit, ax[0], zorder=6, y_scale=y_scale[0])

    if fit.galaxy.photometry_exists:
        add_photometry_posterior(fit, ax[-1], zorder=2, y_scale=y_scale[-1])

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_fit.pdf"
        plt.savefig(plotpath, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax


def add_photometry_posterior(fit, ax, zorder=4, y_scale=None):

    mask = (fit.galaxy.photometry[:, 1] > 0.)
    upper_lims = fit.galaxy.photometry[:, 1] + fit.galaxy.photometry[:, 2]
    ymax = 1.05*np.max(upper_lims[mask])

    if not y_scale:
        y_scale = int(np.log10(ymax))-1

    # Calculate posterior median redshift.
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])

    else:
        redshift = fit.fitted_model.model_components["redshift"]

    # Plot the posterior photometry and full spectrum.
    log_wavs = np.log10(fit.posterior.model_galaxy.wavelengths*(1.+redshift))
    log_eff_wavs = np.log10(fit.galaxy.filter_set.eff_wavs)

    spec_post = np.percentile(fit.posterior.samples["spectrum_full"],
                              (16, 84), axis=0).T*10**-y_scale

    spec_post = spec_post.astype(float) #fixes weird isfinite error

    ax.plot(log_wavs, spec_post[:, 0], color="navajowhite", zorder=zorder-1)
    ax.plot(log_wavs, spec_post[:, 1], color="navajowhite", zorder=zorder-1)

    ax.fill_between(log_wavs, spec_post[:, 0], spec_post[:, 1],
                    zorder=zorder-1, color="navajowhite", linewidth=0)


    phot_post = np.percentile(fit.posterior.samples["photometry"],
                              (16, 84), axis=0).T

    for j in range(fit.galaxy.photometry.shape[0]):
        phot_band = fit.posterior.samples["photometry"][:, j]
        mask = (phot_band > phot_post[j, 0]) & (phot_band < phot_post[j, 1])
        phot_1sig = phot_band[mask]*10**-y_scale
        wav_array = np.zeros(phot_1sig.shape[0]) + log_eff_wavs[j]

        if phot_1sig.min() < ymax*10**-y_scale:
            ax.scatter(wav_array, phot_1sig, color="darkorange",
                       zorder=zorder, alpha=0.05, s=100, rasterized=True)


def add_spectrum_posterior(fit, ax, zorder=4, y_scale=None):

    ymax = 1.05*np.max(fit.galaxy.spectrum[:, 1])

    if not y_scale:
        y_scale = int(np.log10(ymax))-1

    wavs = fit.galaxy.spectrum[:, 0]
    spec_post = np.copy(fit.posterior.samples["spectrum"])

    if "calib" in list(fit.posterior.samples):
        spec_post /= fit.posterior.samples["calib"]

    if "noise" in list(fit.posterior.samples):
        spec_post += fit.posterior.samples["noise"]

    post = np.percentile(spec_post, (16, 50, 84), axis=0).T*10**-y_scale

    ax.plot(wavs, post[:, 1], color="sandybrown", zorder=zorder, lw=1.5)
    ax.fill_between(wavs, post[:, 0], post[:, 2], color="sandybrown",
                    zorder=zorder, alpha=0.75, linewidth=0)

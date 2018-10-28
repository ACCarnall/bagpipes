from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *


def plot_1d_posterior(fit, fit2=False, show=False, save=True):

    update_rcParams()

    # Generate list of parameters to plot
    sfh_names = ["stellar_mass", "sfr", "ssfr", "tform"]
    names = sfh_names + fit.fitted_model.params

    # Get axis labels
    labels = fix_param_names(names)

    # Set up the figure using gridspec
    n_plots = len(names)
    n_rows = int(n_plots//4)

    if n_plots % 4:
        n_rows += 1

    fig = plt.figure(figsize=(12, 3*n_rows))
    gs = mpl.gridspec.GridSpec(n_rows, 4, hspace=0.4, wspace=0.2)

    # Add axes to figure
    axes = []
    for i in range(n_rows):
        for j in range(4):
            if 4*i + (j+1) <= n_plots:
                axes.append(plt.subplot(gs[i, j]))
                plt.setp(axes[-1].get_yticklabels(), visible=False)

    #
    for i in range(len(names)):
        name = names[i]
        label = labels[i]
        samples = np.copy(fit.posterior.samples[name])

        if fit2 and name in fit2.fit_params + sfh_names:
            extra_samples = np.copy(fit2.posterior.samples[name])

        # Log parameter samples and labels for parameters with log priors
        if (i > 4 and fit.fitted_model.pdfs[i-4] == "log_10") or name == "sfr":
            samples = np.log10(samples)

            if fit2 and name in fit2.fit_params + sfh_names:
                extra_samples = np.log10(extra_samples)

            if tex_on:
                label = "$\\mathrm{log_{10}}(" + label[1:-1] + ")$"

            else:
                label = "log_10(" + label + ")"

        try:
            hist1d(samples[np.invert(np.isnan(samples))], axes[i],
                   smooth=True, percentiles=not fit2)

        except ValueError:
            pass

        if fit2 and name in fit2.fit_params + sfh_names:
            hist1d(extra_samples, axes[i], smooth=True, color="purple",
                   percentiles=False, zorder=2)

            low = np.max([np.min([samples.min(), extra_samples.min()]), -99.])
            high = np.max([samples.max(), extra_samples.max()])
            axes[i].set_xlim(low, high)

        axes[i].set_xlabel(label)
        auto_x_ticks(axes[i], nticks=3)

    if save:
        path = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_1d_post.pdf"

        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, axes

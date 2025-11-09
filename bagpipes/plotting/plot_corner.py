from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import corner
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *


def plot_corner(fit, show=False, save=True, bins=25, type="fit_params"):
    """ Make a corner plot of the fitted parameters. """

    update_rcParams()

    names = fit.fitted_model.params
    samples = np.copy(fit.posterior.samples2d)

    # Set up axis labels
    if tex_on:
        labels = fix_param_names(names)

    else:
        labels = fit.fitted_model.params.copy()

    # Log any parameters with log_10 priors to make them easier to see
    for i in range(fit.fitted_model.ndim):
        if fit.fitted_model.pdfs[i] == "log_10":
            samples[:, i] = np.log10(samples[:, i])

            if tex_on:
                labels[i] = "$\\mathrm{log_{10}}(" + labels[i][1:-1] + ")$"

            else:
                labels[i] = "log_10(" + labels[i] + ")"

    # Replace any r parameters for Dirichlet distributions with t_x vals
    j = 0
    for i in range(fit.fitted_model.ndim):
        if "dirichlet" in fit.fitted_model.params[i]:
            comp = fit.fitted_model.params[i].split(":")[0]
            n_x = fit.fitted_model.model_components[comp]["bins"]
            t_percentile = int(np.round(100*(j+1)/n_x))

            samples[:, i] = fit.posterior.samples[comp + ":tx"][:, j]
            j += 1

            if tex_on:
                labels[i] = "$t_{" + str(t_percentile) + "}\ /\ \mathrm{Gyr}$"

            else:
                labels[i] = "t" + str(t_percentile) + " / Gyr"

    # Make the corner plot
    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 13},
                        smooth=1., smooth1d=1., bins=bins)

    # Save the corner plot to file
    if save:
        plotpath = ("pipes/plots/" + fit.run + "/" + fit.galaxy.ID
                    + "_corner.pdf")

        plt.savefig(plotpath, bbox_inches="tight")
        plt.close(fig)

    # Alternatively show the corner plot
    if show:
        plt.show()
        plt.close(fig)

    return fig

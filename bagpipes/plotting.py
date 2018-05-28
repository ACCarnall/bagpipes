from __future__ import print_function, division, absolute_import

import numpy as np
import corner
import copy

from distutils.spawn import find_executable
from scipy.ndimage import gaussian_filter

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    if find_executable("latex"):
        mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        mpl.rc('text', usetex=True)

        tex_on = True
        mpl.rcParams["text.usetex"] = True

    else:
        tex_on = False
        mpl.rcParams["text.usetex"] = False

except ImportError:
    print("Bagpipes: Matplotlib import failed, plotting unavailable.")

from . import utils

mpl.rcParams["lines.linewidth"] = 2.
mpl.rcParams["axes.linewidth"] = 1.5
mpl.rcParams["axes.labelsize"] = 18.
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["ytick.direction"] = "in"

latex_names = {"redshift": "z",
               "metallicity": "Z",
               "massformed": "\\mathrm{log_{10}(M",
               "mass": "\\mathrm{log_{10}(M_*",
               "tau": "\\tau",
               "alpha": "\\alpha",
               "beta": "\\beta",
               "age": "a",
               "Av": "{A_V}",
               "veldisp": "\\sigma_{vel}",
               "0": "\\mathrm{N}0", "1": "\\mathrm{N}1", "2": "\\mathrm{N}2",
               "3": "\\mathrm{N}3", "4": "\\mathrm{N}4", "5": "\\mathrm{N}5",
               "6": "\\mathrm{N}6", "7": "\\mathrm{N}7", "8": "\\mathrm{N}8",
               "9": "\\mathrm{N}9", "10": "\\mathrm{N}10",
               "hypspec": "\\mathcal{H}_\\mathrm{spec}",
               "hypphot": "\\mathcal{H}_\\mathrm{phot}",
               "sfr": "\\mathrm{SFR}",
               "mwa": "\\mathrm{a_{mw}}",
               "tmw": "\\mathrm{t_{form}}",
               "ssfr": "\\mathrm{log_{10}(sSFR",
               }

latex_units = {"metallicity": "Z_{\\odot}",
               "massformed": "M_{\\odot})}",
               "mass": "M_{\\odot})}",
               "tau": "\\mathrm{Gyr}",
               "age": "\\mathrm{Gyr}",
               "Av": "\\mathrm{mag}",
               "veldisp": "\\mathrm{km/s}",
               "sfr": "\\mathrm{M_\\odot\\ yr}^{-1}",
               "ssfr": "\\mathrm{yr}^{-1})}",
               "mwa": "\\mathrm{Gyr}",
               "tmw": "\\mathrm{Gyr}"
               }

latex_comps = {"dblplaw": "dpl",
               "exponential": "exp",
               "constant": "const",
               "delayed": "del",
               "polynomial": "poly",
               "nebular": "neb",
               "lognormal": "lnorm"
               }

""" Plot functions are used by the other classes to generate quick-look
plots. Add functions are used to add details to a passed axis, thus they
can be used to create a variety of plots either by the plot functions
or by the user. """


def plot_sfh(sfh, show=True, style="smooth"):
    """ Make a quick plot of an individual sfh. """
    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, 1, 1)

    add_sfh(sfh, ax, style=style)

    if show:
        plt.show()
        plt.close(fig)

    return fig


def plot_model_galaxy(model, show=True):
    """ Make a quick plot of an individual model galaxy. """
    naxes = 1

    if (model.filt_list is not None and model.spec_wavs is not None):
        naxes = 2

    fig = plt.figure(figsize=(12, 4.*naxes))
    gs = mpl.gridspec.GridSpec(naxes, 1, hspace=0.3)

    if model.spec_wavs is not None:
        spec_ax = plt.subplot(gs[0, 0])
        add_spectrum(model.spectrum, spec_ax,
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


def plot_galaxy(galaxy, show=True, polynomial=None):
    """ Make a quick plot of the data loaded into a galaxy object. """
    naxes = 1

    if (galaxy.photometry_exists and galaxy.spectrum_exists):
        naxes = 2

    fig = plt.figure(figsize=(12, 4.*naxes))
    gs = mpl.gridspec.GridSpec(naxes, 1, hspace=0.3)

    if galaxy.spectrum_exists:
        spec_ax = plt.subplot(gs[0, 0])
        plot_spec = np.copy(galaxy.spectrum)

        if polynomial is not None:
            plot_spec[:, 1] /= polynomial

        add_spectrum(plot_spec, spec_ax)
        if galaxy.photometry_exists:
            add_observed_photometry_linear(galaxy, spec_ax)
        axes = [spec_ax]

    if galaxy.photometry_exists and galaxy.spectrum_exists:
        phot_ax = plt.subplot(gs[1, 0])
        add_observed_photometry(galaxy, phot_ax)
        axes.append(phot_ax)

    elif galaxy.photometry_exists:
        phot_ax = plt.subplot(gs[0, 0])
        add_observed_photometry(galaxy, phot_ax)
        axes = [phot_ax]

    if show:
        plt.show()
        plt.close(fig)

    return fig, axes


def plot_fit(fit, show=False, save=True):
    """ Plot the observational data and posterior from a fit object. """

    if "polynomial" in list(fit.posterior):
        median_poly = np.median(fit.posterior["polynomial"], axis=0)
        fig, axes = plot_galaxy(fit.galaxy, show=False, polynomial=median_poly)

    else:
        fig, axes = plot_galaxy(fit.galaxy, show=False)

    if fit.galaxy.spectrum_exists:
        add_spectrum_posterior(fit, axes[0], zorder=6)

    if fit.galaxy.photometry_exists and fit.galaxy.spectrum_exists:
        add_photometry_posterior(fit, axes[1], zorder=2)

    elif fit.galaxy.photometry_exists:
        add_photometry_posterior(fit, axes[0], zorder=2)

    if save:
        plotpath = ("pipes/plots/" + fit.run + "/" + fit.galaxy.ID
                    + "_fit.pdf")

        plt.savefig(plotpath, bbox_inches="tight")

    if show:
        plt.show()
        plt.close(fig)

    return fig, axes


def plot_sfh_post(fit, show=True):
        fig = plt.figure(figsize=(12, 4))
        ax = plt.subplot(1, 1, 1)

        add_sfh_posterior(fit, ax, style="smooth")

        if show:
            plt.show()
            plt.close(fig)

        return fig, ax


def add_sfh(sfh, ax, zorder=4, style="smooth"):
    """ Creates a plot of sfr(t) for a given star-formation history. """

    # Plot the sfh.
    if style in ["smooth", "both"]:
        sfr = sfh.sfr["total"]
        ax.plot((sfh.age_of_universe - sfh.ages)*10**-9, sfr, color="black")

    if style in ["step", "both"]:
        sfr = sfh.weights["total"]/utils.chosen_age_widths
        sfh_x, sfh_y = make_hist_arrays(utils.chosen_age_lhs, sfr)
        ax.plot((sfh.age_of_universe - sfh_x)*10**-9, sfh_y, color="black")

    # Set limits.
    ax.set_xlim(sfh.age_of_universe*10**-9, 0.)
    ax.set_ylim(bottom=0.)

    ax2 = ax.twiny()
    ax2.set_xticks(np.interp([0, 0.5, 1, 2, 4, 10], utils.z_array,
                   utils.age_at_z))
    ax2.set_xticklabels(["$0$", "$0.5$", "$1$", "$2$", "$4$", "$10$"])
    ax2.set_xlim(ax.get_xlim())

    # Add labels.
    if tex_on:
        ax.set_ylabel("$\\mathrm{SFR\\ /\\ M_\\odot\\ \\mathrm{yr}^{-1}}$")
        ax.set_xlabel("$\\mathrm{Age\\ of\\ Universe\\ /\\ Gyr}$")
        ax2.set_xlabel("$\\mathrm{Redshift}$")

    else:
        ax.set_ylabel("SFR / M_sol yr^-1")
        ax.set_xlabel("Age of Universe / Gyr")
        ax2.set_xlabel("Redshift")


def add_spectrum(spectrum, ax, x_ticks=None, zorder=4, z_non_zero=True):
    """ Add a spectrum to the passed axes. Adds errors if they are
    included in the spectrum object as a third column. """

    # Sort out axis limits
    ymax = 1.05*np.max(spectrum[:, 1])

    y_scale = int(np.log10(ymax))-1

    ax.set_ylim(0., ymax*10**-y_scale)
    ax.set_xlim(spectrum[0, 0], spectrum[-1, 0])

    # Plot the data
    if spectrum.shape[1] == 2:
        ax.plot(spectrum[:, 0], spectrum[:, 1]*10**-y_scale,
                color="sandybrown", zorder=zorder)

    elif spectrum.shape[1] == 3:
        ax.plot(spectrum[:, 0], spectrum[:, 1]*10**-y_scale,
                color="dodgerblue", zorder=zorder, lw=1)

        ax.fill_between(spectrum[:, 0],
                        (spectrum[:, 1] - spectrum[:, 2])*10**-y_scale,
                        (spectrum[:, 1] + spectrum[:, 2])*10**-y_scale,
                        color="dodgerblue", zorder=zorder-1, alpha=0.75,
                        linewidth=0)

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    # Sort out axis labels.
    auto_axis_label(ax, y_scale, z_non_zero=z_non_zero)

    return ax


def add_model_photometry(model, ax, x_ticks=None, zorder=4):
    """ Adds model photometry to the passed axis. """

    # Sort out axis limits
    xmin = np.log10(model.eff_wavs[0])-0.025
    xmax = np.log10(model.eff_wavs[-1])+0.025
    ax.set_xlim(xmin, xmax)

    redshifted_wavs = model.chosen_wavs*(1.+model.model_comp["redshift"])

    spec_mask = ((redshifted_wavs > 10**xmin) & (redshifted_wavs < 10**xmax))

    ymax = 1.05*np.max(model.spectrum_full[spec_mask])

    y_scale = int(np.log10(ymax))-1

    ax.set_ylim(0., ymax*10**-y_scale)

    # Plot the data
    ax.plot(np.log10(redshifted_wavs),
            model.spectrum_full*10**-y_scale, color="navajowhite",
            zorder=zorder-1)

    ax.scatter(np.log10(model.eff_wavs), model.photometry*10**-y_scale,
               color="darkorange", s=150, zorder=zorder)

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


def add_observed_photometry(galaxy, ax, x_ticks=None, zorder=4):
    """ Adds photometric data to the passed axes. """

    # Sort out axis limits
    ax.set_xlim((np.log10(galaxy.eff_wavs[0])-0.025),
                (np.log10(galaxy.eff_wavs[-1])+0.025))

    mask = (galaxy.photometry[:, 1] > 0.)
    ymax = 1.05*np.max((galaxy.photometry[:, 1]+galaxy.photometry[:, 2])[mask])

    y_scale = int(np.log10(ymax))-1

    ax.set_ylim(0., ymax*10**-y_scale)

    # Plot the data
    ax.errorbar(np.log10(galaxy.photometry[:, 0]),
                galaxy.photometry[:, 1]*10**-y_scale,
                yerr=galaxy.photometry[:, 2]*10**-y_scale, lw=1.0,
                linestyle=" ", capsize=3, capthick=1, zorder=zorder-1,
                color="black")

    ax.scatter(np.log10(galaxy.photometry[:, 0]),
               galaxy.photometry[:, 1]*10**-y_scale, color="blue", s=75,
               zorder=zorder, linewidth=1, facecolor="blue", edgecolor="black")

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    auto_axis_label(ax, y_scale, log_x=True)

    return ax


def add_observed_photometry_linear(galaxy, ax, zorder=4):
    """ Adds photometric data to the passed axes without doing any
    manipulation of the axes or labels. """

    ymax = 1.05*np.max(galaxy.photometry[:, 1]+galaxy.photometry[:, 2])

    y_scale = int(np.log10(ymax))-1

    # Plot the data
    ax.errorbar(galaxy.photometry[:, 0],
                galaxy.photometry[:, 1]*10**-y_scale,
                yerr=galaxy.photometry[:, 2]*10**-y_scale, lw=1.0,
                linestyle=" ", capsize=3, capthick=1, zorder=zorder-1,
                color="black")

    ax.scatter(galaxy.photometry[:, 0],
               galaxy.photometry[:, 1]*10**-y_scale, color="blue", s=75,
               zorder=zorder, linewidth=1, facecolor="blue", edgecolor="black")

    return ax


def add_photometry_posterior(fit, ax, zorder=4):

    mask = (fit.galaxy.photometry[:, 1] > 0.)
    ymax = 1.05*np.max((fit.galaxy.photometry[:, 1]
                        + fit.galaxy.photometry[:, 2])[mask])

    y_scale = int(np.log10(ymax))-1

    if "redshift" in list(fit.posterior):
        redshift = fit.posterior["median"]["redshift"]

    else:
        redshift = fit.fixed_values[fit.fixed_params.index("redshift")]

    spec_full_post = fit.posterior["spectrum_full"]
    phot_post = fit.posterior["photometry"]
    log_wavs = np.log10(fit.model.chosen_wavs*(1.+redshift))
    log_eff_wavs = np.log10(fit.model.eff_wavs)

    spec_full_low = np.percentile(spec_full_post, 16, axis=0)*10**-y_scale
    spec_full_med = np.percentile(spec_full_post, 50, axis=0)*10**-y_scale
    spec_full_high = np.percentile(spec_full_post, 84, axis=0)*10**-y_scale

    ax.plot(log_wavs, spec_full_low, color="navajowhite", zorder=zorder-1)
    ax.plot(log_wavs, spec_full_high, color="navajowhite", zorder=zorder-1)
    ax.fill_between(log_wavs, spec_full_low, spec_full_high, zorder=zorder-1,
                    color="navajowhite", linewidth=0)

    phot_low = np.percentile(fit.posterior["photometry"], 16, axis=0)
    phot_high = np.percentile(fit.posterior["photometry"], 84, axis=0)

    for j in range(fit.model.photometry.shape[0]):
        phot_band = fit.posterior["photometry"][:, j]
        mask = (phot_band > phot_low[j]) & (phot_band < phot_high[j])
        phot_1sig = phot_band[mask]*10**-y_scale
        wav_array = np.zeros(phot_1sig.shape[0]) + log_eff_wavs[j]
        ax.scatter(wav_array, phot_1sig, color="darkorange",
                   zorder=zorder, alpha=0.05, s=100, rasterized=True)


def add_spectrum_posterior(fit, ax, zorder=4):

    ymax = 1.05*np.max(fit.galaxy.spectrum[:, 1])

    y_scale = int(np.log10(ymax))-1

    wavs = fit.model.spectrum[:, 0]

    spec_post = fit.posterior["spectrum"]

    if "polynomial" in list(fit.posterior):
        spec_post /= fit.posterior["polynomial"]

    spec_low = np.percentile(spec_post, 16, axis=0)*10**-y_scale
    spec_med = np.percentile(spec_post, 50, axis=0)*10**-y_scale
    spec_high = np.percentile(spec_post, 84, axis=0)*10**-y_scale

    ax.plot(wavs, spec_med, color="sandybrown", zorder=zorder, lw=1.5)
    ax.fill_between(wavs, spec_low, spec_high, color="sandybrown",
                    zorder=zorder, alpha=0.75, linewidth=0)


def add_sfh_posterior(fit, ax, style="smooth", colorscheme="bw", variable="sfr"):

    color1 = "black"
    color2 = "gray"

    if colorscheme == "irnbru":
        color1 = "darkorange"
        color2 = "navajowhite"

    sfh_post = fit.posterior["sfh"]

    # Calculate median redshift and median age of Universe
    if "redshift" in list(fit.posterior):
        redshift = fit.posterior["median"]["redshift"]

    else:
        redshift = fit.fixed_values[fit.fixed_params.index("redshift")]

    age_of_universe = np.interp(redshift, utils.z_array, utils.age_at_z)

    # set plot_post to the relevant grid of posterior quantities.
    if variable == "sfr":
        if style == "smooth":
            plot_post = sfh_post
            x = fit.model.sfh.ages
    """
        elif style="step":
            plot_post = np.zeros((plot_post.shape[0],
                                  2*plot_post.shape[1]))

            for j in range(plot_post.shape[0]):
                x, plot_post[j, :] = make_hist_arrays(utils.chosen_age_lhs,
                                                      plot_post[j, :])
    
        if variable == "ssfr":
        ssfr_post = np.zeros_like(self.posterior["sfh"])

        for i in range(self.posterior["sfh"].shape[1]):

            for j in range(chosen_ages.shape[0]):

                if np.sum(self.posterior["sfh"][j:, i]) != 0.:
                    ssfr_post[j, i] = np.log10(sfh_post[j, i]
                                               / np.sum(sfh_post[j:, i]
                                               * utils.chosen_age_widths[j:]))

                else:
                    ssfr_post[j, i] = 0.

        plot_post = ssfr_post
    """
    # Change the x and y values to allow plotting as a histogram instead
    # of a smooth function.

    post_low = np.percentile(plot_post, 16, axis=0)
    post_med = np.percentile(plot_post, 50, axis=0)
    post_high = np.percentile(plot_post, 84, axis=0)

    # Plot the SFH
    x = age_of_universe - x*10**-9

    ax.plot(x, post_high, color=color2, zorder=5)
    ax.plot(x, post_med, color=color1, zorder=6)
    ax.plot(x, post_low, color=color2, zorder=5)
    ax.fill_between(x, post_low, post_high, color=color2,
                    alpha=0.75, zorder=4)

    if variable == "sfr":
        ax.set_ylim(0., np.max([ax.get_ylim()[1], 1.1*np.max(post_high)]))

    elif variable == "ssfr":
        ax.set_ylim(-12.5, -7.5)

    ax.set_xlim(age_of_universe, 0)

    ax2 = ax.twiny()
    ax2.set_xticks(np.interp([0, 0.5, 1, 2, 4, 10], utils.z_array,
                   utils.age_at_z))
    ax2.set_xticklabels(["$0$", "$0.5$", "$1$", "$2$", "$4$", "$10$"])
    ax2.set_xlim(ax.get_xlim())

    if tex_on:
        if variable == "sfr":
            ax.set_ylabel("$\\mathrm{SFR\\ /\\ M_\\odot\\ \\mathrm{yr}^{-1}}$")

        elif variable == "ssfr":
            ax.set_ylabel("$\\mathrm{sSFR\\ /\\ \\mathrm{yr}^{-1}}$")

        ax.set_xlabel("$\\mathrm{Age\\ of\\ Universe\\ /\\ Gyr}$")
        ax2.set_xlabel("$\\mathrm{Redshift}$")

    else:
        if variable == "sfr":
            ax.set_ylabel("SFR / M_sol yr^-1")

        elif variable == "ssfr":
            ax.set_ylabel("sSFR / yr^-1")

        ax.set_xlabel("Age of Universe / Gyr")
        ax2.set_xlabel("Redshift")


def plot_poly(fit, style="percentiles", show=True):
    """ Plot the posterior of the polynomial spectral correction. """

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    wavs = fit.model.spectrum[:, 0]
    poly_post = np.ones_like(fit.posterior["polynomial"]).astype(float)
    poly_post /= fit.posterior["polynomial"]

    if style == "individual":
        for i in range(poly_post.shape[0]):
            plt.plot(wavs, poly_post[i, :], color="gray", alpha=0.05)

    elif style == "percentiles":

        poly_post_low = np.percentile(poly_post, 16, axis=0)
        poly_post_med = np.percentile(poly_post, 50, axis=0)
        poly_post_high = np.percentile(poly_post, 84, axis=0)

        ax.plot(wavs, poly_post_low, color="navajowhite", zorder=10)
        ax.plot(wavs, poly_post_med, color="darkorange", zorder=10)
        ax.plot(wavs, poly_post_high, color="navajowhite", zorder=10)
        ax.fill_between(wavs, poly_post_low, poly_post_high, lw=0,
                        color="navajowhite", alpha=0.75, zorder=9)

    ax.set_xlim(wavs[0], wavs[-1])

    if tex_on:
        ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
        ax.set_ylabel("$\\mathrm{Spectrum\\ multiplied\\ by}$")

    else:
        ax.set_xlabel("lambda / A")
        ax.set_ylabel("Spectrum multiplied by")

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax


def fix_param_names(fit_params):
    new_params = []

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

    return new_params


def plot_corner(fit, show=False, save=True):
    """ Make a corner plot of the fitted parameters. """

    samples = copy.copy(fit.posterior["samples"])

    if tex_on:
        labels = fix_param_names(fit.fit_params)

    else:
        labels = fit.fit_params

    for i in range(fit.ndim):
        if fit.priors[i] == "log_10":
            samples[:, i] = np.log10(samples[:, i])

            if tex_on:
                labels[i] = "$\\mathrm{log_{10}}(" + labels[i][1:-1] + ")$"

            else:
                labels[i] = "log_10(" + labels[i] + ")"

    fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_kwargs={"fontsize": 13},
                        smooth=2., smooth1d=1.5, bins=50)

    sfh_ax = fig.add_axes([0.65, 0.59, 0.32, 0.15], zorder=10)
    sfr_ax = fig.add_axes([0.82, 0.82, 0.15, 0.15], zorder=10)
    tmw_ax = fig.add_axes([0.65, 0.82, 0.15, 0.15], zorder=10)

    add_sfh_posterior(fit, sfh_ax)
    hist1d(fit.posterior["tmw"], tmw_ax)
    hist1d(fit.posterior["sfr"], sfr_ax)

    sfr_ax.set_xlabel("$\\mathrm{SFR\\ /\\ M_\\odot\\ \\mathrm{yr}^{-1}}$")

    tmw_ax.set_xlabel("$t_\\mathrm{form}\\ /\\ \\mathrm{Gyr}$")

    fig.text(0.725, 0.978, "$t_\\mathrm{form}\\ /\\ \\mathrm{Gyr}$ = $"
             + str(np.round(np.percentile(fit.posterior["tmw"], 50), 2))
             + "^{+" + str(np.round(np.percentile(fit.posterior["tmw"], 84)
                           - np.percentile(fit.posterior["tmw"], 50), 2))
             + "}_{-" + str(np.round(np.percentile(fit.posterior["tmw"], 50)
                            - np.percentile(fit.posterior["tmw"], 16), 2))
             + "}$",
             horizontalalignment="center", size=14)

    fig.text(0.895, 0.978,
             "$\\mathrm{SFR\\ /\\ M_\\odot\\ \\mathrm{yr}^{-1}}$ = $"
             + str(np.round(np.percentile(fit.posterior["sfr"], 50), 2))
             + "^{+" + str(np.round(np.percentile(fit.posterior["sfr"], 84)
                           - np.percentile(fit.posterior["sfr"], 50), 2))
             + "}_{-" + str(np.round(np.percentile(fit.posterior["sfr"], 50)
                            - np.percentile(fit.posterior["sfr"], 16), 2))
             + "}$",
             horizontalalignment="center", size=14)

    if save:
        plotpath = ("pipes/plots/" + fit.run + "/" + fit.galaxy.ID
                    + "_corner.pdf")

        plt.savefig(plotpath, bbox_inches="tight")

    if show:
        plt.show()
        plt.close(fig)

    return fig


def plot_1d_posterior(fit, show=False, save=True):

    post_quantities = fit.fit_params
    n_plots = len(post_quantities)
    n_rows = int(n_plots//4) + 1

    fig = plt.figure(figsize=(12, 3*n_rows))
    gs = mpl.gridspec.GridSpec(n_rows, 4, hspace=0.4, wspace=0.2)

    axes = []
    for i in range(n_rows):
        for j in range(4):
            if 4*i + (j+1) <= n_plots:
                axes.append(plt.subplot(gs[i, j]))
                plt.setp(axes[-1].get_yticklabels(), visible=False)

    labels = fix_param_names(post_quantities)

    for i in range(len(post_quantities)):
        samples = fit.posterior[post_quantities[i]]

        if i < fit.ndim and fit.priors[i] == "log_10":
            samples = np.log10(samples)

            if tex_on:
                labels[i] = "$\\mathrm{log_{10}}(" + labels[i][1:-1] + ")$"

            else:
                labels[i] = "log_10(" + labels[i] + ")"

        hist1d(samples, axes[i], smooth=True)
        axes[i].set_xlabel(labels[i])
        x_range = samples.max() - samples.min()
        auto_x_ticks(axes[i], nticks=3)

    if save:
        plotpath = ("pipes/plots/" + fit.run + "/" + fit.galaxy.ID
                    + "_1d_posterior.pdf")

        plt.savefig(plotpath, bbox_inches="tight")

    if show:
        plt.show()
        plt.close(fig)

    return fig


""" Extra ancilliay functions. """


def hist1d(samples, ax, smooth=False, label=None):

    if label is not None:
        x_label = fix_param_names([label])
        ax.set_xlabel(x_label[0])


    y, x = np.histogram(samples, bins=50, range=(samples.min(), samples.max()))
    y = gaussian_filter(y, 1.5)

    if smooth:
        ax.plot((x[:-1] + x[1:])/2., y, color="darkorange")
        ax.fill_between((x[:-1] + x[1:])/2., np.zeros_like(y), y,
                        color="navajowhite", alpha=0.75)

    else:
        x_hist, y_hist = make_hist_arrays(x, y)
        ax.plot(x_hist, y_hist, color="black")

    for percentile in [16, 50, 84]:
        ax.axvline(np.percentile(samples, percentile), linestyle="--",
                   color="black")

    ax.set_ylim(bottom=0)
    ax.set_xlim(samples.min(), samples.max())
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
            ax.set_xlabel("$\\mathrm{log_{10}}\\Big(\\lambda / \\mathrm{\\AA}"
                          + "\\Big)$")

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


def make_hist_arrays(x, y):
    """ convert x and y arrays for a line plot to a histogram plot. """
    hist_x = np.array(zip(x[:-1], x[1:])).flatten()
    hist_y = np.array(zip(y, y)).flatten()

    return hist_x, hist_y

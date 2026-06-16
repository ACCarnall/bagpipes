from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *
from .plot_galaxy import plot_galaxy, add_observed_photometry, add_observed_photometry_linear
from .plot_spectrum import add_spectrum


def plot_spectrum_posterior(fit, figsize=None, show=False, save=True):
    """ 
    Plots the fitted spectrum plot, including the input and posterior fitted spectrum, residuals and fitted GP noise (if applicable),
    and photometry

    Parameters
    ----------
    fit : object
        The bagpipes.fitting.fit object
    figsize : tuple
        Size of the figure
    show : bool
        Whether to show the figure during runtime.
    save : bool
        Whether to save the resulting figure. Save path is ./pipes/plots/[runID]/[galID]_fit.pdf
    """

    fit.posterior.get_advanced_quantities()

    update_rcParams()

    # check if AGN component is used in the fit, if yes, calculate the AGN spectrum
    _cal_agn_spec(fit)

    # sort out how many panels are needed
    # if only fitting photometry, there will be 1 panel
    # if only fitting spectroscopy, there will be at least 2 panels: main panel and residual panel
    # if correlated noise is turned on, add one additional panel to total 3
    # if there is both spectroscopy and photometry, there will be 3 panels if without correlated noise, 4 if with
    # if there is both spec and photo, and calib module is turned on, there will be 4 panels if without correlated noise, 5 if with
    photo_panel = fit.galaxy.photometry_exists
    spec_panel = fit.galaxy.spectrum_exists
    noise_panel = 'noise' in fit.posterior.samples.keys()
    calib_panel = 'calib' in fit.posterior.samples.keys()

    if tex_on:
        wavelength_label = "$\\lambda / \\mathrm{\\AA}$"
        noise_label = "$P(\\Phi)$"
    else:
        wavelength_label = "lambda / A"
        noise_label = "P(Phi)"

    # Make the figure
    axes = []
    if photo_panel and not spec_panel:
        # 1 panel, photo only
        add_photometry_posterior(fit, ax[-1], zorder=2, y_scale=y_scale[-1])

        if save:
            plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_fit.pdf"
            plt.savefig(plotpath, bbox_inches="tight")
            plt.close(fig)

        if show:
            plt.show()
            plt.close(fig)

        return fig, ax

    elif not photo_panel and not noise_panel and not calib_panel:
        # 2 panels, spec only
        if figsize is None:
            figsize = (15, 1.2*4)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 1, hspace=0., wspace=0.)
        ax_spec = plt.subplot(gs[:3])
        ax_res = plt.subplot(gs[3])
        axes = [ax_spec, ax_res]

    elif not photo_panel and noise_panel and not calib_panel:
        # 3 panels, spec only
        if figsize is None:
            figsize = (15, 1.2*5)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(5, 1, hspace=0., wspace=0.)
        ax_spec = plt.subplot(gs[:3])
        ax_res = plt.subplot(gs[3])
        ax_noise = plt.subplot(gs[4])
        axes = [ax_spec, ax_res, ax_noise]

    elif photo_panel and not noise_panel and not calib_panel:
        # 3 panels, spec + phot
        if figsize is None:
            figsize = (15, 1.2*8)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(8, 1, hspace=0., wspace=0.)
        ax_spec = plt.subplot(gs[:3])
        ax_res = plt.subplot(gs[3])
        ax_phot = plt.subplot(gs[5:])
        axes = [ax_spec, ax_res, ax_phot]

    elif photo_panel and noise_panel and not calib_panel:
        # 4 panels, spec + phot
        if figsize is None:
            figsize = (15, 1.2*9)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(9, 1, hspace=0., wspace=0.)
        ax_spec = plt.subplot(gs[:3])
        ax_res = plt.subplot(gs[3])
        ax_noise = plt.subplot(gs[4])
        ax_phot = plt.subplot(gs[6:])
        axes = [ax_spec, ax_res, ax_noise, ax_phot]

    elif photo_panel and not noise_panel and calib_panel:
        # 4 panels, spec + phot
        if figsize is None:
            figsize = (15, 1.2*9)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(9, 1, hspace=0., wspace=0.)
        ax_spec = plt.subplot(gs[:3])
        ax_res = plt.subplot(gs[3])
        ax_calib = plt.subplot(gs[4])
        ax_phot = plt.subplot(gs[6:])
        axes = [ax_spec, ax_res, ax_calib, ax_phot]

    elif photo_panel and noise_panel and calib_panel:
        # 5 panels, spec + phot
        if figsize is None:
            figsize = (15, 1.2*10)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(10, 1, hspace=0., wspace=0.)
        ax_spec = plt.subplot(gs[:3])
        ax_res = plt.subplot(gs[3])
        ax_noise = plt.subplot(gs[4])
        ax_calib = plt.subplot(gs[5])
        ax_phot = plt.subplot(gs[7:])
        axes = [ax_spec, ax_res, ax_noise, ax_calib, ax_phot]

    else:
        raise ValueError("figure creation failed, number of panels undetermined.")
    
    # if the code gets to this point, there must be a spec panel

    # find wavelength indices that were masked by setting uncertaintites to very high in input data
    spec_mask_ind = np.where(fit.galaxy.spectrum[:,2]>1)[0]
    # obtain an uncertainty spectrum where those intentionally masked are set to nan
    obs_noise_nan = fit.galaxy.spectrum[:,2].copy()
    obs_noise_nan[obs_noise_nan>9e10] = np.nan

    # First plot the observational data
    y_scale_spec = add_spectrum(fit.galaxy.spectrum, ax_spec, label='fitted obs spec')
    # plot main median fitted spec line
    add_spectrum_posterior(fit, ax_spec, y_scale=y_scale_spec, zorder=6)
    # plot line for physical model spectrum
    if noise_panel or calib_panel:
        add_physical_spec(ax_spec, fit, y_scale_spec, color='magenta')
    
    if photo_panel:
        # plot observed photometry
        _ = add_observed_photometry_linear(fit.galaxy, ax_spec, y_scale=y_scale_spec)
        y_scale_phot = add_observed_photometry(fit.galaxy, ax_phot)
        # plot fitted photometry
        add_photometry_posterior(fit, ax_phot, zorder=2, y_scale=y_scale_phot)
    
    # get non masked fluxes to adjust y limits
    non_masked_obs_spec = np.delete(fit.galaxy.spectrum, spec_mask_ind, axis=0)
    # fix y limits
    if ax_spec.get_ylim()[0] < 0.9*min(non_masked_obs_spec[:,1])*10**-y_scale_spec:
        ax_spec.set_ylim(bottom=0.9*min(non_masked_obs_spec[:,1])*10**-y_scale_spec)
    if ax_spec.get_ylim()[1] > 1.1*max(non_masked_obs_spec[:,1])*10**-y_scale_spec:
        ax_spec.set_ylim(top=1.1*max(non_masked_obs_spec[:,1])*10**-y_scale_spec)
        
    # plot AGN component if it is included in the fit
    if 'agn' in fit.fit_instructions:
        # set the ylim of ax_spec to start from 0, so we can plot the agn component near 0
        ax_spec.set_ylim(bottom=0)

        agn_percentiles = np.percentile(fit.posterior.samples['agn_spectrum'],(16,50,84),axis=0)*10**-y_scale_spec
        ax_spec.plot(fit.galaxy.spectrum[:,0], agn_percentiles[1],color="maroon", zorder=5)
        ax_spec.fill_between(fit.galaxy.spectrum[:,0], agn_percentiles[0], agn_percentiles[2],
                             color='pink', alpha=0.7, zorder=4)

    # calculate residuals
    residuals = get_residual_spec(fit, y_scale_spec)
    # plot residuals
    non_masked_res = np.delete(residuals, spec_mask_ind)
    ax_res.axhline(0, color="black", ls="--", lw=1)
    ax_res.plot(np.delete(fit.galaxy.spectrum[:,0], spec_mask_ind), non_masked_res, color="sandybrown", zorder=1)
             
    ax_res.set_ylabel('residual')
    ax_res.set_ylim([1.1*min(non_masked_res), 1.1*max(non_masked_res)])

    spec_xlim = [min(fit.galaxy.spectrum[:,0]),max(fit.galaxy.spectrum[:,0])]
    ax_spec.set_xlim(spec_xlim)
    ax_res.set_xlim(ax_spec.get_xlim())

    # Plot the noise factor
    if noise_panel:
        ax_noise.axhline(0, color="black", ls="--", lw=1)
        
        noise_percentiles = np.percentile(fit.posterior.samples['noise'],(16,50,84),axis=0)*10**-y_scale_spec
        ax_noise.plot(fit.galaxy.spectrum[:,0], noise_percentiles[1],color="sandybrown", zorder=1)
        ax_noise.fill_between(fit.galaxy.spectrum[:,0], noise_percentiles[0], noise_percentiles[2],
                         color='navajowhite', zorder=-1)
        ax_noise.set_xlim(ax_spec.get_xlim())
        auto_x_ticks(ax_spec)
        ax_noise.set_xlabel(wavelength_label)
        ax_noise.set_ylabel('noise')

    # plot the calib factor
    if calib_panel:
        ax_calib.axhline(1, color="black", ls="--", lw=1)

        calib_percentiles = np.percentile(fit.posterior.samples['calib'],(16,50,84),axis=0)
        ax_calib.plot(fit.galaxy.spectrum[:,0], calib_percentiles[1],color="sandybrown", zorder=1)
        ax_calib.fill_between(fit.galaxy.spectrum[:,0], calib_percentiles[0], calib_percentiles[2],
                         color='navajowhite', zorder=-1)
        ax_calib.set_xlim(ax_spec.get_xlim())
        auto_x_ticks(ax_spec)
        ax_calib.set_xlabel(wavelength_label)
        ax_calib.set_ylabel(r'$P(\Phi)$')

    #recover masks on spectrum and plot them as gray bands in residual and other plots
    if len(spec_mask_ind) > 0:
        mask_edges = get_mask_edges(spec_mask_ind)
        ylim = ax_res.get_ylim()
        draw_vertical_mask_regions(ax_res, fit.galaxy.spectrum, mask_edges, limits=[-10,10])
        ax_res.set_ylim(ylim)
        if noise_panel:
            ylim = ax_noise.get_ylim()
            draw_vertical_mask_regions(ax_noise, fit.galaxy.spectrum, mask_edges, limits=[-10,10])
            ax_noise.set_ylim(ylim)
        if calib_panel:
            ylim = ax_calib.get_ylim()
            draw_vertical_mask_regions(ax_calib, fit.galaxy.spectrum, mask_edges, limits=[-10,10])
            ax_calib.set_ylim(ylim)

    # plot observational uncertainty along residuals and noise
    add_obs_unc(fit, ax_res, obs_noise_nan, y_scale_spec, freeze_ylims=True)
    if noise_panel:
        add_obs_unc(fit, ax_noise, obs_noise_nan, y_scale_spec, freeze_ylims=True)

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_fit.pdf"
        plt.savefig(plotpath, bbox_inches="tight")
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)
    
    return fig, axes


def add_photometry_posterior(fit, ax, zorder=4, y_scale=None, color1=None,
                             color2=None, skip_no_obs=False,
                             background_spectrum=True, label=None):

    if color1 == None:
        color1 = "navajowhite"

    if color2 == None:
        color2 = "darkorange"

    mask = (fit.galaxy.photometry[:, 1] > 0.)
    upper_lims = fit.galaxy.photometry[:, 1] + fit.galaxy.photometry[:, 2]
    ymax = 1.05*np.max(upper_lims[mask])

    if not y_scale:
        y_scale = float(int(np.log10(ymax))-1)

    # Calculate posterior median redshift.
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])

    else:
        redshift = fit.fitted_model.model_components["redshift"]

    # Plot the posterior photometry and full spectrum.
    log_wavs = np.log10(fit.posterior.model_galaxy.wavelengths*(1.+redshift))
    log_eff_wavs = np.log10(fit.galaxy.filter_set.eff_wavs)

    if background_spectrum:
        spec_post = np.percentile(fit.posterior.samples["spectrum_full"],
                                  (16, 84), axis=0).T*10**-y_scale

        spec_post = spec_post.astype(float)  # fixes weird isfinite error

        ax.plot(log_wavs, spec_post[:, 0], color=color1,
                zorder=zorder-1, label=label)

        ax.plot(log_wavs, spec_post[:, 1], color=color1,
                zorder=zorder-1)

        ax.fill_between(log_wavs, spec_post[:, 0], spec_post[:, 1],
                        zorder=zorder-1, color=color1, linewidth=0)

    phot_post = np.percentile(fit.posterior.samples["photometry"],
                              (16, 84), axis=0).T

    for j in range(fit.galaxy.photometry.shape[0]):

        if skip_no_obs and fit.galaxy.photometry[j, 1] == 0.:
            continue

        phot_band = fit.posterior.samples["photometry"][:, j]
        mask = (phot_band > phot_post[j, 0]) & (phot_band < phot_post[j, 1])
        phot_1sig = phot_band[mask]*10**-y_scale
        wav_array = np.zeros(phot_1sig.shape[0]) + log_eff_wavs[j]

        if phot_1sig.min() < ymax*10**-y_scale:
            ax.scatter(wav_array, phot_1sig, color=color2,
                       zorder=zorder, alpha=0.05, s=100, rasterized=True)

def add_spectrum_posterior(fit, ax, zorder=4, y_scale=None):

    ymax = 1.05*np.max(fit.galaxy.spectrum[:, 1])

    if not y_scale:
        y_scale = float(int(np.log10(ymax))-1)

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

def get_mask_edges(spec_mask_ind):
    # get the edges in wavelength for each mask in the spectrum,
    # used for making vertical gray bands in panels
    mask_edges = [[spec_mask_ind[0]],[]]
    for i,indi in enumerate(spec_mask_ind[:-1]):
        if spec_mask_ind[i+1] - indi > 1:
            mask_edges[1].append(indi)
            mask_edges[0].append(spec_mask_ind[i+1])
    mask_edges[1].append(spec_mask_ind[-1])
    mask_edges = np.array(mask_edges).T
    
    return mask_edges
    
def draw_vertical_mask_regions(ax, spectrum, mask_edges, limits=[-1,1], color='lightgray'):
    # draw the vertical gray bands in panels for spec masks
    for [mask_min, mask_max] in mask_edges:
        ax.fill_between([spectrum[:,0][mask_min], spectrum[:,0][mask_max]],
                         [limits[0]]*2, [limits[1]]*2, color=color, zorder=2)
                         
def get_residual_spec(fit, y_scale):
    # calculate the residual spectrum
    spec_post = np.copy(fit.posterior.samples["spectrum"])
    
    if "calib" in fit.posterior.samples.keys():
        spec_post /= fit.posterior.samples["calib"]
    if "noise" in fit.posterior.samples.keys():
        spec_post += fit.posterior.samples["noise"]
        
    post_median = np.median(spec_post, axis=0)

    residuals = (fit.galaxy.spectrum[:,1] - post_median)*10**-y_scale
    
    return residuals

def add_physical_spec(ax, fit, y_scale=None, color='magenta', lw=1.0, label=None):
    ymax = 1.05*np.max(fit.galaxy.spectrum[:, 1])

    if not y_scale:
        y_scale = float(int(np.log10(ymax))-1)

    post_physical_median = np.median(fit.posterior.samples["spectrum"], axis=0)
    ax.plot(fit.galaxy.spectrum[:, 0], post_physical_median*10**-y_scale, color=color, lw=lw, zorder=4, label=label)

def add_obs_unc(fit, ax, obs_noise, y_scale, freeze_ylims=False):
    """ adds observational uncertainty lines to the residual and noise panels. Also checks if it has GP:scaling free parameter, if it has, also adds in a scaled obs unc version """
    ax.plot(fit.galaxy.spectrum[:,0], obs_noise*10**-y_scale,
            color='steelblue', lw=1, zorder=4, label='obs noise')
    ylims = ax.get_ylim()
    if 'noise:scaling' in fit.posterior.samples.keys():
        median_noise_scale = np.median(fit.posterior.samples['noise:scaling'])
        ax.plot(fit.galaxy.spectrum[:,0], obs_noise*median_noise_scale*10**-y_scale,
                color='cyan', lw=1, zorder=4, ls='--', label='scaled obs noise')
    if freeze_ylims:
        ax.set_ylim(ylims)

# make better plot_spec with AGN component as share axis in the top panel but with same scale
def _cal_agn_spec(fit):
    # check if AGN component is used in the fit, if yes, calculate the AGN spectrum
    if 'agn' in fit.fit_instructions:
        agn_spectrum = np.zeros((fit.posterior.n_samples, len(fit.galaxy.spec_wavs)))
        m = fit.posterior.fitted_model.model_galaxy
        agn_comp = m.model_comp["agn"].copy()
        
        for i in range(fit.posterior.n_samples):
            # build the agn model comp dictionary
            for param in fit.fitted_model.params:
                if param.split(':')[0] == 'agn':
                    agn_comp[param.split(':')[1]] = fit.posterior.samples[param][i]
                    
            m.agn.update(agn_comp)

            agn_spec = m.agn.spectrum
            agn_spec *= m.igm.trans(fit.posterior.samples["redshift"][i])

            # interpolate raw AGN spectrum onto observed wavelength grid
            zplus1 = (fit.posterior.samples["redshift"][i] + 1.)
            agn_interp = np.interp(m.spec_wavs, m.agn.wavelengths*zplus1,
                                    agn_spec/zplus1, left=0, right=0)
            agn_spectrum[i,:] = agn_interp

        # add AGN spectrum to posterior samples
        fit.posterior.samples['agn_spectrum'] = agn_spectrum

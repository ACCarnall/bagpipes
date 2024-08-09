from __future__ import print_function, division, absolute_import

import numpy as np
import os
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *

from .. import utils

try:
    use_bpass = bool(int(os.environ['use_bpass']))
    print('use_bpass: ',bool(int(os.environ['use_bpass'])))
except KeyError:
    use_bpass = False

if use_bpass:
    print('Setup to use BPASS')
    from .. import config_bpass as config
else:
    print('Setup to use BC03')
    from .. import config


def plot_sfh_posterior(fit, show=False, save=True, colorscheme="bw", zvals=[0, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 25]):
    """ Make a plot of the SFH posterior. """

    update_rcParams()

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot()

    add_sfh_posterior(fit, ax, colorscheme=colorscheme, zvals=zvals, save = save)

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_sfh.pdf"
        plt.savefig(plotpath, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax


def add_sfh_posterior(fit, ax, colorscheme="bw", z_axis=True, zorder=4, alpha=0.6, plottype='absolute', 
                      label=None, zvals=[0, 0.5, 1, 2, 4, 10], color='black', use_color=False, timescale='Gyr', save = True, return_sfh = False):

    color1 = "black"
    color2 = "gray"


    if colorscheme == "irnbru":
        color1 = "darkorange"
        color2 = "navajowhite"
        alpha = 0.6

    if colorscheme == "purple":
        color1 = "purple"
        color2 = "purple"
        alpha = 0.4

    if colorscheme == "blue":
        color1 = "dodgerblue"
        color2 = "dodgerblue"
        alpha = 0.7
    
    if use_color:
        color1 = color
        color2 = color
        alpha = alpha 
    # Calculate median redshift and median age of Universe
    if "redshift" in fit.fitted_model.params:
        redshift = np.median(fit.posterior.samples["redshift"])

    else:
        redshift = fit.fitted_model.model_components["redshift"]
   
    if timescale == 'Gyr':
        factor = 10**-9
    elif timescale == 'Myr':
        factor = 10**-6
    elif timescale == 'yr':
        factor = 1
    else:
        raise ValueError("Unit must be Gyr, Myr or yr")
    
    age_of_universe = np.interp(redshift, utils.z_array, utils.age_at_z) *factor/(10**-9)

    # Calculate median and confidence interval for SFH posterior
    post = np.percentile(fit.posterior.samples["sfh"], (16, 50, 84), axis=0).T

    if plottype == 'lookback':
        x = fit.posterior.sfh.ages*factor
    elif plottype == 'absolute':
        # Plot the SFH
        x = age_of_universe - fit.posterior.sfh.ages*factor
    

    ax.plot(x, post[:, 1], color=color1, zorder=zorder+1, lw=1, alpha=1, label=label)
    ax.fill_between(x, post[:, 0], post[:, 2], color=color2,
                    alpha=alpha, zorder=zorder, lw=0)

   
    timescales_sfr = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1] # Gyr
    results = []
    results_err_low = []
    results_err_high = []
    for timescale_sfr in timescales_sfr:
    
        mask = (x[0] - x < timescale_sfr) & (x > 0)
        av_sfr = np.median(post[mask, 1])
        av_sfr_err_low = np.median(post[mask, 1] - post[mask, 0])
        av_sfr_err_high = np.median(post[mask, 2] - post[mask, 1])
        results.append(av_sfr)
        results_err_low.append(av_sfr_err_low)
        results_err_high.append(av_sfr_err_high)
    
    if not os.path.exists("pipes/sfr/" + fit.run):
        os.makedirs("pipes/sfr/" + fit.run)
    if save:
        np.savetxt("pipes/sfr/" + fit.run + "/" + fit.galaxy.ID + "_sfh_timescales.txt", np.array([timescales_sfr, results, results_err_low, results_err_high]).T, header="timescale(Gyr), av_sfr, av_sfr_err_low, av_sfr_err_high")


    if plottype == 'absolute':
        ax.set_xlim(age_of_universe, 0)
        ax.set_ylim(0., np.max([ax.get_ylim()[1], 1.1*np.max(post[:, 2])]))

        
    elif plottype == 'lookback':
       
        if ax.get_xlim()[0] < 0:
            ax.set_xlim(0, age_of_universe*1.1)
        else:
            ax.set_xlim(0, np.max([age_of_universe*1.1, ax.get_xlim()[1]]))
        ax.set_ylim(1e-2, np.max([ax.get_ylim()[1], 1.1*np.max(post[:, 2])]))
        
    # Add redshift axis along the top
    if z_axis:
        if plottype == 'absolute':
            ax2 = add_z_axis(ax, zvals=zvals)
      
    # Set axis labels
    #if tex_on:
    #ax.set_ylabel("$\\mathrm{SFR\\ /\\ M_\\odot\\ \\mathrm{yr}^{-1}}$")
            
    ax.set_ylabel("$\\mathbf{\\mathrm{SFR\\ (M_\\odot\\ \\mathrm{yr}^{-1}})}$", fontsize='small')
            
    if plottype == 'absolute':
        ax.set_xlabel(f"$\\mathbf{{\\mathrm{{Age\\ of\\ Universe \\ ({timescale})}}}}$", fontsize='small')

    else:
        ax.set_xlabel(f"$\\mathbf{{\\mathrm{{Lookback\\ Time \\ ({timescale})}}}}$", fontsize='small')

    if return_sfh:
        return x, post
    
    #
    #else:
    #    ax.set_ylabel("SFR / Msol/yr")
    #    ax.set_xlabel("Age of Universe / Gyr")

from __future__ import print_function, division, absolute_import
import os 
import numpy as np
import time
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


try:
    from tqdm import tqdm
except:
    def tqdm(x):
        return x

from scipy.interpolate import RegularGridInterpolator


def plot_csfh_posterior(fit, show=False, save=True, colorscheme="bw", zvals=[0, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 25]):

    update_rcParams()

    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot()

    add_csfh_posterior(fit, ax, colorscheme=colorscheme, zvals=zvals)

    if save:
        plotpath = "pipes/plots/" + fit.run + "/" + fit.galaxy.ID + "_csfh.pdf"
        #print(plotpath)
        plt.savefig(plotpath, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)

    if show:
        plt.show()
        plt.close(fig)

    return fig, ax



def add_csfh_posterior(fit, ax, colorscheme="bw", z_axis=True, zorder=4, alpha=0.6, plottype='absolute',
                      label=None, zvals=[0, 0.5, 1, 2, 4, 10], color='black', use_color=False, timescale='Gyr', debug = False):

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



    times = (fit.posterior.sfh.age_of_universe - fit.posterior.sfh.ages)*10**-9
    print(fit.posterior.sfh.age_of_universe, age_of_universe)
    times_reduced = times[times >= 0]
    
    file = h5py.File(fit.fname[:-1] + ".h5", "a")
    if 'mah_grid' in file.keys():
        mah_grid = file['mah_grid'][:]
        file.close()
    else:

        mah_grid = np.zeros((fit.n_posterior, times.shape[0]))
        grid = RegularGridInterpolator((config.metallicities, config.age_sampling), fit.posterior.sfh.live_frac_grid, fill_value=0.7, bounds_error=False)
        if debug:
            print('Age of Universe', fit.posterior.sfh.age_of_universe/1e9)
            print('Times', np.min(times), np.max(times))

            print('Range of grid:')
            print('Metallicity:', np.min(config.metallicities), np.max(config.metallicities))
            print('Ages:', np.min(config.age_sampling), np.max(config.age_sampling))

        mah_grid = optimize_mah_grid2(fit, config, grid)
        file.create_dataset('mah_grid', data=mah_grid, compression='gzip', dtype = np.float32)
        file.close()
    
    masses = np.nanpercentile(mah_grid, (2.5, 16, 50, 84, 97.5), axis=0).T


    if z_axis:
        if plottype == 'absolute':
            ax2 = add_z_axis(ax, zvals=zvals, reverse = True)

    if plottype == 'absolute':
        ax.plot(times, masses[:, 2], color=color1, zorder=zorder, label=label)
        ax.fill_between(times, masses[:, 1], masses[:, 3], color=color2, alpha=alpha, zorder=zorder)
        ax.fill_between(times, masses[:, 0], masses[:, 4], color=color2, alpha=alpha/2, zorder=zorder)
        ax.set_xlim(age_of_universe, 0)

    elif plottype == 'lookback':
        ax.plot(age_of_universe - times_reduced, masses[times >= 0, 2], color=color1, zorder=zorder, label=label)
        ax.fill_between(age_of_universe - times_reduced, masses[times >= 0, 1], masses[times >= 0, 3], color=color2, alpha=alpha, zorder=zorder)
        ax.fill_between(age_of_universe - times_reduced, masses[times >= 0, 0], masses[times >= 0, 4], color=color2, alpha=alpha/2, zorder=zorder)
 
    ax.set_ylabel(r"$\mathrm{Stellar\,Mass\,(\log_{10}\,M_{\odot}/M_{\star})}$", fontsize='medium')

    if plottype == 'absolute':
        ax.set_xlabel(f"$\\mathbf{{\\mathrm{{Age\\ of\\ Universe \\ ({timescale})}}}}$", fontsize='medium')

    else:
        ax.set_xlabel(f"$\\mathbf{{\\mathrm{{Lookback\\ Time \\ ({timescale})}}}}$", fontsize='medium')
    ax.set_yscale('log')

    ax.set_ylim(1e2, 1.1*np.nanmax(masses[:, 3]))

    #ax.set_ylim(0, 1.1*np.nanmax(masses[:, 3]))

import numpy as np
from tqdm import tqdm

def optimize_mah_grid(fit, config, grid):
    n_posterior = fit.n_posterior
    n_times = len(fit.posterior.sfh.ages)
    mah_grid = np.zeros((n_posterior, n_times))
    
    mass_weighted_zmet = np.clip(fit.posterior.samples["mass_weighted_zmet"], 
                                 np.min(config.metallicities), 
                                 np.max(config.metallicities))
    
    # Pre-compute all age differences
    age_diffs = np.diff(fit.posterior.sfh.ages)
    
    for k in tqdm(range(n_posterior)):
        # Initialize with the full array for the first step
        alive_frac = grid((mass_weighted_zmet[k], fit.posterior.sfh.ages - fit.posterior.sfh.ages[0]))
        mah_grid[k, 0] = np.sum(fit.posterior.samples["sfh"][k] * fit.posterior.sfh.age_widths * alive_frac)
        
        for l in range(1, n_times):
            # Shift alive_frac array and compute only the new value
            alive_frac = np.roll(alive_frac, -1)
            alive_frac[-1] = grid((mass_weighted_zmet[k], np.array([0])))
            
            # Update mah_grid using the shifted alive_frac
            mah_grid[k, l] = np.sum(fit.posterior.samples["sfh"][k, l:] * 
                                    fit.posterior.sfh.age_widths[l:] * 
                                    alive_frac[:-l])
    
    return mah_grid



def optimize_mah_grid2(fit, config, grid):
    '''Fastest version of the MAH grid calculation so far'''
    n_posterior = fit.n_posterior
    n_times = len(fit.posterior.sfh.ages)
    mah_grid = np.zeros((n_posterior, n_times))
    
    mass_weighted_zmet = np.clip(fit.posterior.samples["mass_weighted_zmet"], 
                                 np.min(config.metallicities), 
                                 np.max(config.metallicities))
    
    # Pre-compute age widths * sfh for all samples
    sfh_weighted = fit.posterior.samples["sfh"] * fit.posterior.sfh.age_widths
    
    for k in tqdm(range(n_posterior)):
        # Initialize with the full array for the first step
        alive_frac = grid((mass_weighted_zmet[k], fit.posterior.sfh.ages - fit.posterior.sfh.ages[0]))
        mah_grid[k, 0] = np.sum(sfh_weighted[k] * alive_frac)
        
        cumsum = np.cumsum(sfh_weighted[k][::-1])[::-1]
        
        for l in range(1, n_times):
            # Compute only the new value
            new_alive_frac = grid((mass_weighted_zmet[k], np.array([0])))
            
            # Update mah_grid using the cumulative sum
            mah_grid[k, l] = cumsum[l] * new_alive_frac + np.sum(
                sfh_weighted[k, l:] * (alive_frac[:-l] - new_alive_frac)
            )
            
            # Update alive_frac for the next iteration
            alive_frac[:-1] = alive_frac[1:]
            alive_frac[-1] = new_alive_frac

    return mah_grid
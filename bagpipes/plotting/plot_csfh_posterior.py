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
    import utils
   
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
    
    mah_grid = np.zeros((fit.n_posterior, times.shape[0]))


    grid = RegularGridInterpolator((config.metallicities, config.age_sampling), fit.posterior.sfh.live_frac_grid, fill_value=0.7, bounds_error=False)
    if debug:
        print('Age of Universe', fit.posterior.sfh.age_of_universe/1e9)
        print('Times', np.min(times), np.max(times))

        print('Range of grid:')
        print('Metallicity:', np.min(config.metallicities), np.max(config.metallicities))
        print('Ages:', np.min(config.age_sampling), np.max(config.age_sampling))

        
    '''
    for k in tqdm(range(fit.n_posterior)):
        mass_weighted_zmet = fit.posterior.samples["mass_weighted_zmet"][k]
        mass_weighted_zmet = np.clip(mass_weighted_zmet, np.min(config.metallicities), np.max(config.metallicities))

        for l in range(times.shape[0]):
            
            ages = (fit.posterior.sfh.ages[l:] - fit.posterior.sfh.ages[l])
            alive_frac = grid((mass_weighted_zmet, ages))
            mah_grid[k, l] = np.sum(fit.posterior.samples["sfh"][k, l:]*fit.posterior.sfh.age_widths[l:] * alive_frac)
           
            # Mass-weighted metallicity at each time step
            #mass_weighted_zmet = np.sum(fit.posterior.sfh.live_frac_grid[:,l:] * fit.posterior.sfh.ceh.grid[:, l:],
            #                                axis=1)
            #mass_weighted_zmet /= np.sum(fit.posterior.sfh.live_frac_grid[:, l:] * fit.posterior.sfh.ceh.grid[:, l:])
            #mass_weighted_zmet *= config.metallicities
            #print(np.shape(mass_weighted_zmet), np.shape(fit.posterior.sfh.ages[l:]))
            #mass_weighted_zmet = np.nansum(mass_weighted_zmet)
            #print(fit.posterior.samples.keys())
            #mass_weighted_zmet = fit.posterior.samples["mass_weighted_zmet"][k]
            # Clip to range of grid
            #mass_weighted_zmet = np.clip(mass_weighted_zmet, np.min(config.metallicities), np.max(config.metallicities))

            # Sum SFH * (1 - return fraction)  * age bin width
            # 1 - return fraction is interpolated onto grid
            # print(fit.posterior.sfh.ages[l:] - fit.posterior.sfh.ages[l], mass_weighted_zmet * np.ones_like(fit.posterior.sfh.ages[l:]))
            #ages = (fit.posterior.sfh.ages[l:] - fit.posterior.sfh.ages[l])
            # Clip to range of grid
            #ages = np.clip(ages, np.min(config.age_sampling), np.max(config.age_sampling))

            #alive_frac = grid((mass_weighted_zmet, ages))
            #alive_frac = 1.0
            #print(np.shape(alive_frac), np.shape(fit.posterior.sfh.age_widths[l:]), np.shape(fit.posterior.samples["sfh"][k, l:]))
            #print(np.min(alive_frac), np.max(alive_frac))
            #print(np.min(ages), np.max(ages))
            
            #print(mass_weighted_zmet, np.shape(alive_frac), print(np.min(ages)), print(np.max(ages)))
            #print(alive_frac)
            #mah_grid[k, l] = np.sum(fit.posterior.samples["sfh"][k, l:]*fit.posterior.sfh.age_widths[l:] * alive_frac)
            
            #*np.interp(fit.posterior.sfh.ages[l:] - fit.posterior.sfh.ages[l],
                                                                                #pipes.config.age_sampling,
                                                                                #fit.posterior.sfh.live_frac_grid[ind, :]))
    
    # Allegedly vectorized equivalent

    mass_weighted_zmet = np.clip(fit.posterior.samples["mass_weighted_zmet"], 
                            np.min(config.metallicities), 
                            np.max(config.metallicities))

    ages = fit.posterior.sfh.ages[np.newaxis, :, np.newaxis] - fit.posterior.sfh.ages[np.newaxis, np.newaxis, :]
    ages = np.maximum(ages, 0)  # Ensure non-negative ages
    #
    # Assuming grid can handle 2D input for metallicities and ages
    stime = time.time()
    alive_frac = grid((mass_weighted_zmet[:, np.newaxis, np.newaxis], ages))
    end1 = time.time() - stime
    print('Time for grid:', end1)
    print(np.shape(alive_frac))
    sfh_expanded = fit.posterior.samples["sfh"][:, np.newaxis, :]
    end2= time.time() - end1
    print('Time for sfh:', end2)
    print(np.shape(sfh_expanded))

    age_widths_expanded = fit.posterior.sfh.age_widths[np.newaxis, np.newaxis, :]
    end3 = time.time() - end2
    print('Time for age widths:', end3)
    print(np.shape(age_widths_expanded))
    new_mah_grid = np.sum(sfh_expanded * age_widths_expanded * alive_frac, axis=2)
    end4 = time.time() - end3
    print('Time for sum:', end4)
    print(np.shape(new_mah_grid))
    print(np.nanmax(mah_grid))
    #assert np.allclose(mah_grid, new_mah_grid)
      print(masses[:,2])
    print('brk')
    print(masses2[:,2])
    end5 = time.time() - end4
    print(end1, end2, end3, end4, end5)
    #print('max', np.max(masses[:, 2]))
    #print('max all', np.max(masses))
    #masses = np.log10(masses)

    
    '''
    mah_grid = optimize_mah_grid2(fit, config, grid)
    masses = np.nanpercentile(mah_grid, (2.5, 16, 50, 84, 97.5), axis=0).T


    if z_axis:
        if plottype == 'absolute':
            ax2 = add_z_axis(ax, zvals=zvals)

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
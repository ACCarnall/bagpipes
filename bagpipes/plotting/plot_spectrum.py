from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

except RuntimeError:
    pass

from .general import *


def add_spectrum(spectrum, ax, x_ticks=None, zorder=4, z_non_zero=True,
                 y_scale=None, ymax=None):
    """ Add a spectrum to the passed axes. Adds errors if they are
    included in the spectrum object as a third column. """

    # Sort out axis limits
    if not ymax:
        ymax = 1.05*np.max(spectrum[:, 1])

    if y_scale is None:
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

        lower = (spectrum[:, 1] - spectrum[:, 2])*10**-y_scale
        upper = (spectrum[:, 1] + spectrum[:, 2])*10**-y_scale

        upper[upper > ymax*10**-y_scale] = ymax*10**-y_scale
        lower[lower < 0.] = 0.

        ax.fill_between(spectrum[:, 0], lower, upper, color="dodgerblue",
                        zorder=zorder-1, alpha=0.75, linewidth=0)

    # Sort out x tick locations
    if x_ticks is None:
        auto_x_ticks(ax)

    else:
        ax.set_xticks(x_ticks)

    # Sort out axis labels.
    auto_axis_label(ax, y_scale, z_non_zero=z_non_zero)

    return y_scale

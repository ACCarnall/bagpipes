from __future__ import print_function, division, absolute_import

import numpy as np
import sys
import os

from astropy.io import fits

""" This code is called once when Bagpipes is first installed in order
to generate the IGM absorption table which is subsequently used for
all IGM calculations. """

path = os.path.dirname(os.path.realpath(__file__)) + "/../grids"
coefs = np.loadtxt(path + "/lyman_series_coefs_inoue_2014_table2.txt")


def get_Inoue14_trans(rest_wavs, z_obs):
    """ Calculate IGM transmission using Inoue et al. (2014) model. """

    if isinstance(rest_wavs, float):
        rest_wavs = np.array([rest_wavs])

    tau_LAF_LS = np.zeros((39, rest_wavs.shape[0]))
    tau_DLA_LS = np.zeros((39, rest_wavs.shape[0]))
    tau_LAF_LC = np.zeros(rest_wavs.shape[0])
    tau_DLA_LC = np.zeros(rest_wavs.shape[0])

    # Populate tau_LAF_LS
    for j in range(39):

        if z_obs < 1.2:
            wav_slice = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
                         & (rest_wavs*(1.+z_obs)
                            < (1+z_obs)*coefs[j, 1]))

            tau_LAF_LS[j, wav_slice] = (coefs[j, 2]
                                        * (rest_wavs[wav_slice]
                                           * (1.+z_obs)/coefs[j, 1])**1.2)

        elif z_obs < 4.7:
            wav_slice_1 = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
                           & (rest_wavs*(1.+z_obs) < 2.2*coefs[j, 1]))
            wav_slice_2 = ((rest_wavs*(1.+z_obs) > 2.2*coefs[j, 1])
                           & (rest_wavs*(1.+z_obs)
                              < (1+z_obs)*coefs[j, 1]))

            tau_LAF_LS[j, wav_slice_1] = (coefs[j, 2]
                                          * (rest_wavs[wav_slice_1]
                                             * (1.+z_obs)/coefs[j, 1])**1.2)

            tau_LAF_LS[j, wav_slice_2] = (coefs[j, 3]
                                          * (rest_wavs[wav_slice_2]
                                             * (1.+z_obs)/coefs[j, 1])**3.7)

        else:
            wav_slice_1 = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
                           & (rest_wavs*(1.+z_obs) < 2.2*coefs[j, 1]))

            wav_slice_2 = ((rest_wavs*(1.+z_obs) > 2.2*coefs[j, 1])
                           & (rest_wavs*(1.+z_obs) < 5.7*coefs[j, 1]))

            wav_slice_3 = ((rest_wavs*(1.+z_obs) > 5.7*coefs[j, 1])
                           & (rest_wavs*(1.+z_obs)
                              < (1+z_obs)*coefs[j, 1]))

            tau_LAF_LS[j, wav_slice_1] = (coefs[j, 2]
                                          * (rest_wavs[wav_slice_1]
                                             * (1.+z_obs)/coefs[j, 1])**1.2)

            tau_LAF_LS[j, wav_slice_2] = (coefs[j, 3]
                                          * (rest_wavs[wav_slice_2]
                                             * (1.+z_obs)/coefs[j, 1])**3.7)

            tau_LAF_LS[j, wav_slice_3] = (coefs[j, 4]
                                          * (rest_wavs[wav_slice_3]
                                             * (1.+z_obs)/coefs[j, 1])**5.5)

    # Populate tau_DLA_LS
    for j in range(39):

        if z_obs < 2.0:
            wav_slice = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
                         & (rest_wavs*(1.+z_obs)
                            < (1+z_obs)*coefs[j, 1]))

            tau_DLA_LS[j, wav_slice] = (coefs[j, 5]
                                        * (rest_wavs[wav_slice]
                                           * (1.+z_obs)/coefs[j, 1])**2.0)

        else:
            wav_slice_1 = ((rest_wavs*(1.+z_obs) > coefs[j, 1])
                           & (rest_wavs*(1.+z_obs) < 3.0*coefs[j, 1]))

            wav_slice_2 = ((rest_wavs*(1.+z_obs) > 3.0*coefs[j, 1])
                           & (rest_wavs*(1.+z_obs) < (1+z_obs)
                              * coefs[j, 1]))

            tau_DLA_LS[j, wav_slice_1] = (coefs[j, 5]
                                          * (rest_wavs[wav_slice_1]
                                             * (1.+z_obs)/coefs[j, 1])**2.0)

            tau_DLA_LS[j, wav_slice_2] = (coefs[j, 6]
                                          * (rest_wavs[wav_slice_2]
                                             * (1.+z_obs)/coefs[j, 1])**3.0)

    # Populate tau_LAF_LC
    if z_obs < 1.2:
        wav_slice = ((rest_wavs*(1.+z_obs) > 911.8)
                     & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

        tau_LAF_LC[wav_slice] = (0.325*((rest_wavs[wav_slice]
                                         * (1.+z_obs)/911.8)**1.2
                                        - (((1+z_obs)**-0.9)
                                           * (rest_wavs[wav_slice]
                                           * (1.+z_obs)/911.8)**2.1)))

    elif z_obs < 4.7:
        wav_slice_1 = ((rest_wavs*(1.+z_obs) > 911.8)
                       & (rest_wavs*(1.+z_obs) < 911.8*2.2))

        wav_slice_2 = ((rest_wavs*(1.+z_obs) > 911.8*2.2)
                       & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

        tau_LAF_LC[wav_slice_1] = (((2.55*10**-2)*((1+z_obs)**1.6)
                                    * (rest_wavs[wav_slice_1]
                                    * (1.+z_obs)/911.8)**2.1)
                                   + (0.325*((rest_wavs[wav_slice_1]
                                      * (1.+z_obs)/911.8)**1.2))
                                   - (0.25*((rest_wavs[wav_slice_1]
                                             * (1.+z_obs)/911.8)**2.1)))

        tau_LAF_LC[wav_slice_2] = ((2.55*10**-2)
                                   * (((1.+z_obs)**1.6)
                                      * ((rest_wavs[wav_slice_2]
                                          * (1.+z_obs)/911.8)**2.1)
                                      - ((rest_wavs[wav_slice_2]
                                          * (1.+z_obs)/911.8)**3.7)))

    else:
        wav_slice_1 = ((rest_wavs*(1.+z_obs) > 911.8)
                       & (rest_wavs*(1.+z_obs) < 911.8*2.2))

        wav_slice_2 = ((rest_wavs*(1.+z_obs) > 911.8*2.2)
                       & (rest_wavs*(1.+z_obs) < 911.8*5.7))

        wav_slice_3 = ((rest_wavs*(1.+z_obs) > 911.8*5.7)
                       & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

        tau_LAF_LC[wav_slice_1] = (((5.22*10**-4)*((1+z_obs)**3.4)
                                    * (rest_wavs[wav_slice_1]
                                       * (1.+z_obs)/911.8)**2.1)
                                   + (0.325*(rest_wavs[wav_slice_1]
                                      * (1.+z_obs)/911.8)**1.2)
                                   - ((3.14*10**-2)*((rest_wavs[wav_slice_1]
                                      * (1.+z_obs)/911.8)**2.1)))

        tau_LAF_LC[wav_slice_2] = (((5.22*10**-4)*((1+z_obs)**3.4)
                                    * (rest_wavs[wav_slice_2]
                                       * (1.+z_obs)/911.8)**2.1)
                                   + (0.218*((rest_wavs[wav_slice_2]
                                             * (1.+z_obs)/911.8)**2.1))
                                   - ((2.55*10**-2)*((rest_wavs[wav_slice_2]
                                                      * (1.+z_obs)
                                                      / 911.8)**3.7)))

        tau_LAF_LC[wav_slice_3] = ((5.22*10**-4)
                                   * (((1+z_obs)**3.4)
                                      * (rest_wavs[wav_slice_3]
                                         * (1.+z_obs)/911.8)**2.1
                                      - (rest_wavs[wav_slice_3]
                                         * (1.+z_obs)/911.8)**5.5))

    # Populate tau_DLA_LC
    if z_obs < 2.0:
        wav_slice = ((rest_wavs*(1.+z_obs) > 911.8)
                     & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

        tau_DLA_LC[wav_slice] = (0.211*((1+z_obs)**2.)
                                 - (7.66*10**-2)*(((1+z_obs)**2.3)
                                                  * (rest_wavs[wav_slice]
                                                     * (1.+z_obs)/911.8)**-0.3)
                                 - 0.135*((rest_wavs[wav_slice]
                                           * (1.+z_obs)/911.8)**2.0))

    else:
        wav_slice_1 = ((rest_wavs*(1.+z_obs) > 911.8)
                       & (rest_wavs*(1.+z_obs) < 911.8*3.0))

        wav_slice_2 = ((rest_wavs*(1.+z_obs) > 911.8*3.0)
                       & (rest_wavs*(1.+z_obs) < 911.8*(1.+z_obs)))

        tau_DLA_LC[wav_slice_1] = (0.634 + (4.7*10**-2)*(1.+z_obs)**3.
                                   - ((1.78*10**-2)*((1.+z_obs)**3.3)
                                      * (rest_wavs[wav_slice_1]
                                         * (1.+z_obs)/911.8)**-0.3)
                                   - (0.135*(rest_wavs[wav_slice_1]
                                             * (1.+z_obs)/911.8)**2.0)
                                   - 0.291*(rest_wavs[wav_slice_1]
                                            * (1.+z_obs)/911.8)**-0.3)

        tau_DLA_LC[wav_slice_2] = ((4.7*10**-2)*(1.+z_obs)**3.
                                   - ((1.78*10**-2)*((1.+z_obs)**3.3)
                                      * (rest_wavs[wav_slice_2]
                                         * (1.+z_obs)/911.8)**-0.3)
                                   - ((2.92*10**-2)
                                      * (rest_wavs[wav_slice_2]
                                         * (1.+z_obs)/911.8)**3.0))

    tau_LAF_LS_sum = np.sum(tau_LAF_LS, axis=0)
    tau_DLA_LS_sum = np.sum(tau_DLA_LS, axis=0)

    tau = tau_LAF_LS_sum + tau_DLA_LS_sum + tau_LAF_LC + tau_DLA_LC

    return np.exp(-tau)


def make_table(z_array, rest_wavs):
    """ Make up the igm absorption table used by bagpipes. """

    print("BAGPIPES: Generating IGM absorption table.")

    d_IGM_grid = np.zeros((z_array.shape[0], rest_wavs.shape[0]))

    for i in range(z_array.shape[0]):
        d_IGM_grid[i, :] = get_Inoue14_trans(rest_wavs, z_array[i])

    hdulist = fits.HDUList(hdus=[fits.PrimaryHDU(),
                                 fits.ImageHDU(name="trans", data=d_IGM_grid),
                                 fits.ImageHDU(name="wavs", data=rest_wavs),
                                 fits.ImageHDU(name="zred", data=z_array)])

    if os.path.exists(path + "/d_igm_grid_inoue14.fits"):
        os.system("rm " + path + "/d_igm_grid_inoue14.fits")

    hdulist.writeto(path + "/d_igm_grid_inoue14.fits")


def test():
    """ Test the above code by generating a plot from Inoue et al.
    (2014). """
    import matplotlib.pyplot as plt
    plt.figure()

    for i in range(2, 7):

        z_obs = float(i)

        rest_wavs = np.arange(0.5, 1500., 0.5)

        trans = get_Inoue14_trans(rest_wavs, z_obs)

        plt.plot(rest_wavs*(1+z_obs), trans, color="black")

    plt.xlim(3000., 9000.)
    plt.ylim(0., 1.)
    plt.xlabel("$\\mathrm{Observed\\ Wavelength\\ (\\AA)}$")
    plt.ylabel("Transmission")
    plt.show()

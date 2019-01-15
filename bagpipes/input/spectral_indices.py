from __future__ import print_function, division, absolute_import

import numpy as np


def measure_index(ind_dict, spectrum, redshift):
    """ Measure spectral indices, either an EW or flux ratio. """

    if spectrum.shape[1] == 3:
        calculate_errs = True

    else:
        calculate_errs = False

    type = ind_dict["type"]
    continuum = ind_dict["continuum"]
    wavs = spectrum[:, 0]/(1. + redshift)

    cont_fluxes = []
    cont_sigs = []

    for i in range(len(continuum)):
        mask = (wavs > continuum[i][0]) & (wavs < continuum[i][1])
        if not np.max(mask):
            raise ValueError("Spectrum does not contain the specified index.")

        slice = spectrum[mask, :]
        cont_fluxes.append(np.mean(slice[:, 1]))
        inv_n = 1./float(slice.shape[0])

        if calculate_errs:
            cont_sigs.append(inv_n*np.sqrt(np.sum(slice[:, 2]**2)))

    if type == "EW":
        feature = ind_dict["feature"]
        feature_mask = (wavs > feature[0]) & (wavs < feature[1])

        if not np.max(feature_mask):
            raise ValueError("Spectrum does not contain the specified index.")

        slice = spectrum[feature_mask, :]
        feature_flux = np.mean(slice[:, 1])
        n_sig = 1./float(slice.shape[0])

        if calculate_errs:
            feature_sig = n_sig*np.sqrt(np.sum(slice[:, 2]**2))

        feature_width = feature[1] - feature[0]

        cont_flux = np.mean(np.array(cont_fluxes))
        ew_num = feature_width*(cont_flux - feature_flux)
        ew = ew_num/cont_flux

        if calculate_errs:
            inv_n = 1./float(len(cont_sigs))
            cont_sig = inv_n*np.sqrt(np.sum(np.array(cont_sigs)**2))

            ew_num_sig = feature_width*np.sqrt(np.sum(np.c_[feature_sig,
                                                            cont_sig]**2))

            ew_sig = np.abs(ew)*np.sqrt(np.sum(np.c_[ew_num_sig/ew_num,
                                                     cont_sig/cont_flux]**2))

            return np.c_[ew, ew_sig]

        return ew

    elif type == "break":
        if not len(continuum) == 2:
            raise ValueError("Two continuum windows are required for"
                             + "measuring a spectral break.")

        ratio = cont_fluxes[1]/cont_fluxes[0]

        if calculate_errs:
            snrs = np.c_[cont_sigs[1]/cont_fluxes[1],
                         cont_sigs[0]/cont_fluxes[0]]

            ratio_sig = np.abs(ratio)*np.sqrt(np.sum(snrs**2))

            return np.c_[ratio, ratio_sig]

        return ratio

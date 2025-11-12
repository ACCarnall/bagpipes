import numpy as np
from astropy.io import fits
import os

# Nebular grids
list_of_hdus_umin_only = [fits.PrimaryHDU()]
list_of_hdus_umin_umax = [fits.PrimaryHDU()]

qpah_fnames = {
"0.10": "smc",
"0.47": "MW3.1_00",
"0.75": "LMC2_00",
"1.12": "MW3.1_10",
"1.49": "LMC2_05",
"1.77": "MW3.1_20",
"2.37": "LMC2_10",
"2.50": "MW3.1_30",
"3.19": "MW3.1_40",
"3.90": "MW3.1_50",
"4.58": "MW3.1_60",
}

qpah_vals_str = ["0.10", "0.47", "0.75", "1.12", "1.49", "1.77", "2.37",
                 "2.50", "3.19", "3.90", "4.58"]

umin_vals_str = ["0.10", "0.15", "0.20", "0.30", "0.40", "0.50", "0.70",
"0.80", "1.00", "1.20", "1.50", "2.00", "2.50", "3.00", "4.00", "5.00",
"7.00", "8.00", "12.0", "15.0", "20.0", "25.0"]



for qpah in qpah_vals_str:

    grid_umin_only = np.zeros((1001, 1+len(umin_vals_str)))
    grid_umin_umax = np.zeros((1001, 1+len(umin_vals_str)))

    grid_umin_only[:,0] = 10**(np.arange(0., 4.001, 0.004) + 4.)
    grid_umin_umax[:,0] = 10**(np.arange(0., 4.001, 0.004) + 4.)

    for i in range(len(umin_vals_str)):

        umin = umin_vals_str[i]

        for skip in [47, 61]:
            try:
                grid_umin_only[:,i+1] = np.loadtxt("../U" + umin + "/U" + umin
                                                   + "_" + umin + "_" + qpah_fnames[qpah]
                                                   + ".txt",
                                                   skiprows=skip, usecols=(1))[::-1]

                grid_umin_umax[:,i+1] = np.loadtxt("../U" + umin + "/U" + umin
                                                   + "_" + "1e6" + "_" + qpah_fnames[qpah]
                                                   + ".txt",
                                                   skiprows=skip, usecols=(1))[::-1]

                grid_umin_only[:,i+1] /= grid_umin_only[:,0]
                grid_umin_umax[:,i+1] /= grid_umin_umax[:,0]

                break

            except:
                pass

    list_of_hdus_umin_only.append(fits.ImageHDU(name="qpah_" + qpah,
                                                data=grid_umin_only))

    list_of_hdus_umin_umax.append(fits.ImageHDU(name="qpah_" + qpah,
                                                data=grid_umin_umax))

hdulist_umin_only = fits.HDUList(hdus=list_of_hdus_umin_only)
hdulist_umin_umax = fits.HDUList(hdus=list_of_hdus_umin_umax)

os.system("rm dl07_grids_umin_only_no_norm.fits")
os.system("rm dl07_grids_umin_umax_no_norm.fits")

hdulist_umin_only.writeto("dl07_grids_umin_only_no_norm.fits")
hdulist_umin_umax.writeto("dl07_grids_umin_umax_no_norm.fits")

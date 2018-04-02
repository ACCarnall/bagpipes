import numpy as np 
from astropy.io import fits
import os

# Nebular grids
list_of_hdus_stellar = [fits.PrimaryHDU()]

zmet_vals = np.array([0.005, 0.02, 0.2, 0.4, 1., 2.5, 5.])

for i in range(zmet_vals.shape[0]):
	list_of_hdus_stellar.append(fits.ImageHDU(name="zmet_" + "%.3f" % zmet_vals[i] + "Zsol", data=np.genfromtxt("bc2003_hr_xmiless_m" + str(i+2) + "2_kroup_ssp.ised_ASCII", skip_header=7, skip_footer=12, usecols=np.arange(1, 13216+1), dtype="float")))

list_of_hdus_stellar.append(fits.ImageHDU(name="liv_mstar_frac", data=np.loadtxt("live_mass_fractions.txt")))

SED_file = open("bc2003_hr_xmiless_m22_kroup_ssp.ised_ASCII")
ages = np.array(SED_file.readline().split(), dtype="float")[1:] #ages[0] = 0.
SED_file.close()

list_of_hdus_stellar.append(fits.ImageHDU(name="Stellar_age_yr", data=ages))

list_of_hdus_stellar.append(fits.ImageHDU(name="wavelengths_AA", data=np.squeeze(np.genfromtxt("bc2003_hr_xmiless_m22_kroup_ssp.ised_ASCII", skip_header = 6, skip_footer=233, usecols=np.arange(1, 13216+1), dtype="float"))))

hdulist_stellar = fits.HDUList(hdus=list_of_hdus_stellar)

os.system("rm bc03_miles_stellar_grids.fits")

hdulist_stellar.writeto("bc03_miles_stellar_grids.fits")

from __future__ import print_function

import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os

import model_manager as models


# N    lambdaj   A_LAF_J_1   A_LAF_J_2   A_LAF_J_3   A_DLA_J_1   A_DLA_J_2

coefs = np.loadtxt(models.install_dir + "/tables/IGM/Lyman_series_coefs_Inoue_2014_Table2.txt")


def get_Inoue14_trans(rest_wavs, z_observed):

	if isinstance(rest_wavs, float):
		rest_wavs = np.array([rest_wavs])

	tau_LAF_LS = np.zeros((39, rest_wavs.shape[0]))
	tau_DLA_LS = np.zeros((39, rest_wavs.shape[0]))
	tau_LAF_LC = np.zeros(rest_wavs.shape[0])
	tau_DLA_LC = np.zeros(rest_wavs.shape[0])

	# Populate tau_LAF_LS
	for j in range(39):

		if z_observed < 1.2:
			wav_slice = (rest_wavs*(1.+z_observed) > coefs[j,1]) & (rest_wavs*(1.+z_observed) < (1+z_observed)*coefs[j,1])

			tau_LAF_LS[j,wav_slice] = coefs[j,2]*(rest_wavs[wav_slice]*(1.+z_observed)/coefs[j,1])**1.2

		elif z_observed < 4.7:
			wav_slice_1 = (rest_wavs*(1.+z_observed) > coefs[j,1]) & (rest_wavs*(1.+z_observed) < 2.2*coefs[j,1])
			wav_slice_2 = (rest_wavs*(1.+z_observed) > 2.2*coefs[j,1]) & (rest_wavs*(1.+z_observed) < (1+z_observed)*coefs[j,1])

			tau_LAF_LS[j,wav_slice_1] = coefs[j,2]*(rest_wavs[wav_slice_1]*(1.+z_observed)/coefs[j,1])**1.2
			tau_LAF_LS[j,wav_slice_2] = coefs[j,3]*(rest_wavs[wav_slice_2]*(1.+z_observed)/coefs[j,1])**3.7


		else:
			wav_slice_1 = (rest_wavs*(1.+z_observed) > coefs[j,1]) & (rest_wavs*(1.+z_observed) < 2.2*coefs[j,1])
			wav_slice_2 = (rest_wavs*(1.+z_observed) > 2.2*coefs[j,1]) & (rest_wavs*(1.+z_observed) < 5.7*coefs[j,1])
			wav_slice_3 = (rest_wavs*(1.+z_observed) > 5.7*coefs[j,1]) & (rest_wavs*(1.+z_observed) < (1+z_observed)*coefs[j,1])

			tau_LAF_LS[j,wav_slice_1] = coefs[j,2]*(rest_wavs[wav_slice_1]*(1.+z_observed)/coefs[j,1])**1.2
			tau_LAF_LS[j,wav_slice_2] = coefs[j,3]*(rest_wavs[wav_slice_2]*(1.+z_observed)/coefs[j,1])**3.7
			tau_LAF_LS[j,wav_slice_3] = coefs[j,4]*(rest_wavs[wav_slice_3]*(1.+z_observed)/coefs[j,1])**5.5


	# Populate tau_DLA_LS
	for j in range(39):

		if z_observed < 2.0:
			wav_slice = (rest_wavs*(1.+z_observed) > coefs[j,1]) & (rest_wavs*(1.+z_observed) < (1+z_observed)*coefs[j,1])

			tau_DLA_LS[j,wav_slice] = coefs[j,5]*(rest_wavs[wav_slice]*(1.+z_observed)/coefs[j,1])**2.0

		else:
			wav_slice_1 = (rest_wavs*(1.+z_observed) > coefs[j,1]) & (rest_wavs*(1.+z_observed) < 3.0*coefs[j,1])
			wav_slice_2 = (rest_wavs*(1.+z_observed) > 3.0*coefs[j,1]) & (rest_wavs*(1.+z_observed) < (1+z_observed)*coefs[j,1])

			tau_DLA_LS[j,wav_slice_1] = coefs[j,5]*(rest_wavs[wav_slice_1]*(1.+z_observed)/coefs[j,1])**2.0
			tau_DLA_LS[j,wav_slice_2] = coefs[j,6]*(rest_wavs[wav_slice_2]*(1.+z_observed)/coefs[j,1])**3.0


	# Populate tau_LAF_LC
	if z_observed < 1.2:
		wav_slice = (rest_wavs*(1.+z_observed) > 911.8) & (rest_wavs*(1.+z_observed) < 911.8*(1.+z_observed))

		tau_LAF_LC[wav_slice] = 0.325*((rest_wavs[wav_slice]*(1.+z_observed)/911.8)**1.2 - ((1+z_observed)**-0.9)*(rest_wavs[wav_slice]*(1.+z_observed)/911.8)**2.1)

	elif z_observed < 4.7:
		wav_slice_1 = (rest_wavs*(1.+z_observed) > 911.8) & (rest_wavs*(1.+z_observed) < 911.8*2.2)
		wav_slice_2 = (rest_wavs*(1.+z_observed) > 911.8*2.2) & (rest_wavs*(1.+z_observed) < 911.8*(1.+z_observed))

		tau_LAF_LC[wav_slice_1] = (2.55*10**-2)*((1+z_observed)**1.6)*(rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**2.1 + 0.325*((rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**1.2) - 0.25*((rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**2.1)
		tau_LAF_LC[wav_slice_2] = (2.55*10**-2)*(((1+z_observed)**1.6)*((rest_wavs[wav_slice_2]*(1.+z_observed)/911.8)**2.1) - ((rest_wavs[wav_slice_2]*(1.+z_observed)/911.8)**3.7))

	else:
		wav_slice_1 = (rest_wavs*(1.+z_observed) > 911.8) & (rest_wavs*(1.+z_observed) < 911.8*2.2)
		wav_slice_2 = (rest_wavs*(1.+z_observed) > 911.8*2.2) & (rest_wavs*(1.+z_observed) < 911.8*5.7)
		wav_slice_3 = (rest_wavs*(1.+z_observed) > 911.8*5.7) & (rest_wavs*(1.+z_observed) < 911.8*(1.+z_observed))

		tau_LAF_LC[wav_slice_1] = (5.22*10**-4)*((1+z_observed)**3.4)*(rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**2.1 + 0.325*((rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**1.2) - (3.14*10**-2)*((rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**2.1)
		tau_LAF_LC[wav_slice_2] = (5.22*10**-4)*((1+z_observed)**3.4)*(rest_wavs[wav_slice_2]*(1.+z_observed)/911.8)**2.1 + 0.218*((rest_wavs[wav_slice_2]*(1.+z_observed)/911.8)**2.1) - (2.55*10**-2)*((rest_wavs[wav_slice_2]*(1.+z_observed)/911.8)**3.7)
		tau_LAF_LC[wav_slice_3] = (5.22*10**-4)*(((1+z_observed)**3.4)*(rest_wavs[wav_slice_3]*(1.+z_observed)/911.8)**2.1 - (rest_wavs[wav_slice_3]*(1.+z_observed)/911.8)**5.5)


	# Populate tau_DLA_LC
	if z_observed < 2.0:
		wav_slice = (rest_wavs*(1.+z_observed) > 911.8) & (rest_wavs*(1.+z_observed) < 911.8*(1.+z_observed))

		tau_DLA_LC[wav_slice] = 0.211*((1+z_observed)**2.) - (7.66*10**-2)*(((1+z_observed)**2.3)*(rest_wavs[wav_slice]*(1.+z_observed)/911.8)**-0.3) - 0.135*((rest_wavs[wav_slice]*(1.+z_observed)/911.8)**2.0)

	else:
		wav_slice_1 = (rest_wavs*(1.+z_observed) > 911.8) & (rest_wavs*(1.+z_observed) < 911.8*3.0)
		wav_slice_2 = (rest_wavs*(1.+z_observed) > 911.8*3.0) & (rest_wavs*(1.+z_observed) < 911.8*(1.+z_observed))

		tau_DLA_LC[wav_slice_1] = 0.634 + (4.7*10**-2)*(1.+z_observed)**3. - (1.78*10**-2)*((1.+z_observed)**3.3)*(rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**-0.3 - 0.135*(rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**2.0 - 0.291*(rest_wavs[wav_slice_1]*(1.+z_observed)/911.8)**-0.3
		tau_DLA_LC[wav_slice_2] = (4.7*10**-2)*(1.+z_observed)**3. - (1.78*10**-2)*((1.+z_observed)**3.3)*(rest_wavs[wav_slice_2]*(1.+z_observed)/911.8)**-0.3 - (2.92*10**-2)*(rest_wavs[wav_slice_2]*(1.+z_observed)/911.8)**3.0

	tau_LAF_LS_sum = np.sum(tau_LAF_LS, axis=0)
	tau_DLA_LS_sum = np.sum(tau_DLA_LS, axis=0)

	tau = tau_LAF_LS_sum + tau_DLA_LS_sum + tau_LAF_LC + tau_DLA_LC

	return np.exp(-tau)



def make_table():
	print("BAGPIPES: Generating IGM absorption table, this may take a few moments the first time you run the code.")

	z_array = np.arange(0.0, 10.01, 0.01)

	rest_wavs = np.arange(1.0, 1225.01, 1.0)

	d_IGM_grid = np.zeros((z_array.shape[0], rest_wavs.shape[0]))

	for i in range(z_array.shape[0]):
		d_IGM_grid[i,:] = get_Inoue14_trans(rest_wavs, z_array[i])

	np.savetxt(models.install_dir + "/tables/IGM//D_IGM_grid_Inoue14.txt", d_IGM_grid)




if __name__ == "__main__":
	plt.figure()

	for i in range(2, 7):

		z_observed = float(i)

		rest_wavs = np.arange(0.5, 1500., 0.5)

		trans = get_Inoue14_trans(rest_wavs, z_observed)

		plt.plot(rest_wavs*(1+z_observed), trans, color="black")


	plt.xlim(3000., 9000.)
	plt.ylim(0., 1.)
	plt.xlabel("$\mathrm{Observed\ Wavelength\ (\AA)}$")
	plt.ylabel("Transmission")
	plt.show()
	

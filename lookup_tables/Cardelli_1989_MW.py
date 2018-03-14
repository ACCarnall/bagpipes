import numpy as np 



def get_cardelli_extinction(wavs):

	A_lambda = np.zeros(wavs.shape)

	inv_mic = 1./(wavs*10.**-4.)

	A_lambda[inv_mic < 1.1] = 0.574*inv_mic[inv_mic < 1.1]**1.61 + (-0.527*inv_mic[inv_mic < 1.1]**1.61)/3.1

	y = inv_mic[(inv_mic > 1.1) & (inv_mic < 3.3)] - 1.82

	A_lambda[(inv_mic > 1.1) & (inv_mic < 3.3)] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7 + (1.41388*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)/3.1

	A_lambda[(inv_mic > 3.3) & (inv_mic < 5.9)] = (1.752 - 0.316*inv_mic[(inv_mic > 3.3) & (inv_mic < 5.9)] - (0.104)/((inv_mic[(inv_mic > 3.3) & (inv_mic < 5.9)]-4.67)**2 + 0.341)) + (-3.09 + 1.825*inv_mic[(inv_mic > 3.3) & (inv_mic < 5.9)] + 1.206/((inv_mic[(inv_mic > 3.3) & (inv_mic < 5.9)]-4.62)**2 + 0.263))/3.1

	A_lambda[(inv_mic > 5.9) & (inv_mic < 8.)] = (1.752 - 0.316*inv_mic[(inv_mic > 5.9) & (inv_mic < 8.)] - (0.104)/((inv_mic[(inv_mic > 5.9) & (inv_mic < 8.)]-4.67)**2 + 0.341) - 0.04473*(inv_mic[(inv_mic > 5.9) & (inv_mic < 8.)] - 5.9)**2 - 0.009779*(inv_mic[(inv_mic > 5.9) & (inv_mic < 8.)]-5.9)**3) + (-3.09 + 1.825*inv_mic[(inv_mic > 5.9) & (inv_mic < 8.)] + 1.206/((inv_mic[(inv_mic > 5.9) & (inv_mic < 8.)]-4.62)**2 + 0.263) + 0.2130*(inv_mic[(inv_mic > 5.9) & (inv_mic < 8.)]-5.9)**2 + 0.1207*(inv_mic[(inv_mic > 5.9) & (inv_mic < 8.)]-5.9)**3)/3.1


	A_lambda[inv_mic > 8.] = -((wavs[inv_mic > 8.] - (10**4/8.)))/250. + (1.752 - 0.316*8. - (0.104)/((8.-4.67)**2 + 0.341) - 0.04473*(8. - 5.9)**2 - 0.009779*(8.-5.9)**3) + (-3.09 + 1.825*8. + 1.206/((8.-4.62)**2 + 0.263) + 0.2130*(8.-5.9)**2 + 0.1207*(8.-5.9)**3)/3.1
	
	return A_lambda



wavs = np.logspace(1., 7., num=10000)


cardelli = np.array([wavs, get_cardelli_extinction(wavs)]).T

np.savetxt("Cardelli_1989_MW.txt", cardelli, header="wavs_angstrom A_lambda/A_V")

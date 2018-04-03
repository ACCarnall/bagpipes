import numpy as np 
import corner
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], "size": 14})
rc('text', usetex=True)


def compare_fits(fit1, fit2, param_names_tolog=[], truths=None, comp_run="."):

	colour1 = "darkorange"
	colour1_2 = "navajowhite"
	colour2 = "purple"
	colour2_2 = "plum"

	alpha1 = 0.75
	alpha2 = 0.85

	params = [param for param in fit1.fit_params if param in fit2.fit_params]

	param_indices1 = []
	param_indices2 = []

	for param in params:
		param_indices1.append(fit1.fit_params.index(param))
		param_indices2.append(fit2.fit_params.index(param))

	ndim = len(params)

	for i in range(ndim):
		if params[i] in param_names_tolog:

			fit1.posterior["samples"][:,param_indices1[i]] = np.log10(fit1.posterior["samples"][:,param_indices1[i]])
			fit2.posterior["samples"][:,param_indices2[i]] = np.log10(fit2.posterior["samples"][:,param_indices2[i]])

	fig = plt.figure(figsize=(ndim*3., ndim*3.))

	gs = gridspec.GridSpec(ndim, ndim)
	gs.update(wspace=0.05, hspace=0.05)

	axes = []

	lims1 = np.zeros((ndim, 2))
	lims2 = np.zeros((ndim, 2))

	for i in range(ndim):
		lims1[i,0] = np.min(fit1.posterior["samples"][:,param_indices1[i]])
		lims1[i,1] = np.max(fit1.posterior["samples"][:,param_indices1[i]])

		lims2[i,0] = np.min(fit2.posterior["samples"][:,param_indices2[i]])
		lims2[i,1] = np.max(fit2.posterior["samples"][:,param_indices2[i]])

	for i in range(ndim):
		for j in range(i+1):
			axes.append(plt.subplot(gs[i,j]))

			axis = axes[-1]

			if i != ndim-1:
				plt.setp(axis.get_xticklabels(), visible=False)

			else:
				axis.set_xlabel(params[j])

			if j != 0 or j == 0 and i == 0:
				plt.setp(axis.get_yticklabels(), visible=False)

			else:
				axis.set_ylabel(params[i])

			if i == j:					
				axis.hist(fit1.posterior["samples"][:,param_indices1[i]], bins=20, color=colour1_2, normed=True, histtype="stepfilled", edgecolor=colour1, range=[np.min([lims1[i,0], lims2[i,0]]), np.max([lims1[i,1], lims2[i,1]])], zorder=6, alpha=alpha1, lw=2)
				axis.hist(fit2.posterior["samples"][:,param_indices2[i]], bins=20, color=colour2_2, normed=True, histtype="stepfilled", edgecolor=colour2, range=[np.min([lims1[i,0], lims2[i,0]]), np.max([lims1[i,1], lims2[i,1]])], zorder=7, alpha=alpha2, lw=2)

				axis.set_xlim(np.min([lims1[i,0], lims2[i,0]]), np.max([lims1[i,1], lims2[i,1]]))

				if truths is not None:
					axis.axvline(truths[i], lw=2, color="#4682b4", zorder=8)

			else:
				corner.hist2d(fit1.posterior["samples"][:,param_indices1[j]], fit1.posterior["samples"][:,param_indices1[i]], ax=axis, smooth=1.5, range=(lims1[j,:], lims1[i,:]), color=colour1)#, plot_density=False)
				corner.hist2d(fit2.posterior["samples"][:,param_indices2[j]], fit2.posterior["samples"][:,param_indices2[i]], ax=axis, smooth=1.5, range=(lims2[j,:], lims2[i,:]), color=colour2)#, plot_density=False)

				axis.set_xlim(np.min([lims1[j,0], lims2[j,0]]), np.max([lims1[j,1], lims2[j,1]]))
				axis.set_ylim(np.min([lims1[i,0], lims2[i,0]]), np.max([lims1[i,1], lims2[i,1]]))

				if truths is not None:
					axis.axhline(truths[i], lw=2, color="#4682b4", zorder=8)
					axis.axvline(truths[j], lw=2, color="#4682b4", zorder=8)
					axis.plot(truths[j], truths[i], "s", color="#4682b4", zorder=8)


		
	sfh_ax = fig.add_axes([0.65, 0.59, 0.32, 0.15], zorder=10)
	sfr_ax = fig.add_axes([0.82, 0.82, 0.15, 0.15], zorder=10)
	tmw_ax = fig.add_axes([0.65, 0.82, 0.15, 0.15], zorder=10)

	
	sfh_x1, sfh_y1, sfh_y_low1, sfh_y_high1 = get_sfh_info(fit1)
	sfh_x2, sfh_y2, sfh_y_low2, sfh_y_high2 = get_sfh_info(fit2)

	# Plot the SFH
	sfh_ax.fill_between(np.interp(fit1.model_components["redshift"], models.z_array, models.age_at_z) - sfh_x1*10**-9, sfh_y_low1, sfh_y_high1, color=colour1_2, alpha=alpha1, lw=2, edgecolor=colour2)
	sfh_ax.plot(np.interp(fit1.model_components["redshift"], models.z_array, models.age_at_z) - sfh_x1*10**-9, sfh_y1, color=colour1, zorder=10)
	sfh_ax.set_xlim(np.interp(fit1.model_components["redshift"], models.z_array, models.age_at_z), 0)

	sfh_ax.fill_between(np.interp(fit2.model_components["redshift"], models.z_array, models.age_at_z) - sfh_x2*10**-9, sfh_y_low2, sfh_y_high2, color=colour2_2, alpha=alpha2, lw=2, edgecolor=colour2)
	sfh_ax.plot(np.interp(fit2.model_components["redshift"], models.z_array, models.age_at_z) - sfh_x2*10**-9, sfh_y2, color=colour2, zorder=10)


	sfh_ax2 = sfh_ax.twiny()
	sfh_ax2.set_xticks(np.interp([0, 0.5, 1, 2, 4, 10], models.z_array, models.age_at_z))
	sfh_ax2.set_xticklabels(["$0$", "$0.5$", "$1$", "$2$", "$4$", "$10$"])
	sfh_ax2.set_xlim(sfh_ax.get_xlim())
	sfh_ax2.set_xlabel("$\mathrm{Redshift}$", size=14)

	# Plot the current star formation rate posterior
	sfr_ax.hist(fit1.posterior["sfr"], bins=20, color=colour1_2, normed=True, histtype="stepfilled", edgecolor=colour1, alpha=alpha1, lw=2, range=(np.min([np.min(fit1.posterior["sfr"]), np.min(fit2.posterior["sfr"])]), np.max([np.max(fit1.posterior["sfr"]), np.max(fit2.posterior["sfr"])])))
	sfr_ax.hist(fit2.posterior["sfr"], bins=20, color=colour2_2, normed=True, histtype="stepfilled", edgecolor=colour2, alpha=alpha2, lw=2, range=(np.min([np.min(fit1.posterior["sfr"]), np.min(fit2.posterior["sfr"])]), np.max([np.max(fit1.posterior["sfr"]), np.max(fit2.posterior["sfr"])])))

	sfr_ax.set_xlabel("$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")
	sfr_ax.set_xlim(np.min([np.min(fit1.posterior["sfr"]), np.min(fit2.posterior["sfr"])]), np.max([np.max(fit1.posterior["sfr"]), np.max(fit2.posterior["sfr"])]))
	sfr_ax.set_yticklabels([])

	# Plot the mass weighted age posterior
	tmw_ax.hist(fit1.posterior["tmw"], bins=20, color=colour1_2, normed=True, histtype="stepfilled", edgecolor=colour1, alpha=alpha1, lw=2, range=(np.min([np.min(fit1.posterior["tmw"]), np.min(fit2.posterior["tmw"])]), np.max([np.max(fit1.posterior["tmw"]), np.max(fit2.posterior["tmw"])])), label="VANDELS")
	tmw_ax.hist(fit2.posterior["tmw"], bins=20, color=colour2_2, normed=True, histtype="stepfilled", edgecolor=colour2, alpha=alpha2, lw=2, range=(np.min([np.min(fit1.posterior["tmw"]), np.min(fit2.posterior["tmw"])]), np.max([np.max(fit1.posterior["tmw"]), np.max(fit2.posterior["tmw"])])), label="Wild")

	tmw_ax.legend(frameon=False, markerfirst=False, fontsize=14, loc=7, ncol=1, bbox_to_anchor=(-0.2, 0.8))

	tmw_ax.set_xlabel("$t(z_\mathrm{form})\ /\ \mathrm{Gyr}$")
	tmw_ax.set_xlim(np.min([np.min(fit1.posterior["tmw"]), np.min(fit2.posterior["tmw"])]), np.max([np.max(fit1.posterior["tmw"]), np.max(fit2.posterior["tmw"])]))
	tmw_ax.set_yticklabels([])

	sfh_ax.set_ylabel("$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$", size=14)
	sfh_ax.set_xlabel("$\mathrm{Age\ of\ Universe\ (Gyr)}$", size=14)
	sfh_ax.set_ylim(0, 1.1*np.max([np.max(sfh_y_high1), np.max(sfh_y_high2)]))
	

	if not os.path.exists(models.install_dir + "/plots/" + comp_run):
		os.mkdir(models.install_dir + "/plots/" + comp_run)

	fig.savefig(models.install_dir + "/plots/" + comp_run + "/" + fit1.Galaxy.ID + "_comp_corner.pdf")#, bbox_inches="tight")



def get_sfh_info(fit):
	# Generate and populate sfh arrays which allow the SFH to be plotted with straight lines across bins of SFH
	sfh_x = np.zeros(2*fit.Model.sfh.ages.shape[0])
	sfh_y = np.zeros(2*fit.Model.sfh.sfr.shape[0])
	sfh_y_low = np.zeros(2*fit.Model.sfh.sfr.shape[0])
	sfh_y_high = np.zeros(2*fit.Model.sfh.sfr.shape[0])

	for j in range(fit.Model.sfh.sfr.shape[0]):

		sfh_x[2*j] = fit.Model.sfh.age_lhs[j]

		sfh_y[2*j] = np.median(fit.posterior["sfh"][:,j])
		sfh_y[2*j + 1] = np.median(fit.posterior["sfh"][:,j])

		sfh_y_low[2*j] = np.percentile(fit.posterior["sfh"][:,j], 16)
		sfh_y_low[2*j + 1] = np.percentile(fit.posterior["sfh"][:,j], 16)

		if sfh_y_low[2*j] < 0:
			sfh_y_low[2*j] = 0.

		if sfh_y_low[2*j+1] < 0:
			sfh_y_low[2*j+1] = 0.

		sfh_y_high[2*j] = np.percentile(fit.posterior["sfh"][:,j], 84)
		sfh_y_high[2*j + 1] = np.percentile(fit.posterior["sfh"][:,j], 84)

		if j == fit.Model.sfh.sfr.shape[0]-1:
			sfh_x[-1] = fit.Model.sfh.age_lhs[-1] + 2*(fit.Model.sfh.ages[-1] - fit.Model.sfh.age_lhs[-1])

		else:
			sfh_x[2*j + 1] = fit.Model.sfh.age_lhs[j+1]

	return sfh_x, sfh_y, sfh_y_low, sfh_y_high





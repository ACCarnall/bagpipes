import numpy as np
rng = np.random.default_rng(seed=9)
from matplotlib import pyplot as plt

import bagpipes as pipes

# Now make a basic fit instructions dictionary.

dust = {}
dust["type"] = "Calzetti"
dust["eta"] = 2.
dust["Av"] = 0.380

nebular = {}
nebular["logU"] = -3.

model_components = {}
model_components["dust"] = dust
model_components["nebular"] = nebular
model_components["t_bc"] = 0.01
model_components["redshift"] = 1.05

continuity = {}
continuity["massformed"] = 11.263
continuity["metallicity"] = 0.936
continuity["metallicity_prior"] = "log_10"
continuity["bin_edges"] = np.arange(0, 5500.0, 1000.0)

def random_SFH(continuity):
    for i in range(1, len(continuity["bin_edges"])-1):
        continuity["dsfr" + str(i)] = rng.standard_t(df=2)
    return continuity

model_components["continuity"] = random_SFH(continuity)

spec_wavs = np.linspace(1000.0, 1e5, 100)
model = pipes.model_galaxy(model_components, spec_wavs=spec_wavs)

fig, ax_SFH = plt.subplots()

ax_SFH.axvline(x=model.sfh.age_of_universe*1e-9, linestyle='--', color='k')

for modeli in range(10):
    # Update model and plot the SFH
    model_components["continuity"] = random_SFH(continuity)
    model.update(model_components)
    ax_SFH.plot((model.sfh.age_of_universe - model.sfh.ages)*1e-9, model.sfh.sfh, alpha=0.8)

ax_SFH.set_xlabel(r"Age of Universe (Gyr)")
ax_SFH.set_ylabel(r"$\mathrm{SFR \, (M_\odot yr^{-1})}$")

ax_SFH.set_xlim(5.8, -0.5)
ax_SFH.set_yscale("log")

fig.savefig("/Users/Joris/Downloads/SFH_mwe.png")
# plt.show()
plt.close()
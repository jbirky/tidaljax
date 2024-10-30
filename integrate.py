# import pdb; pdb.set_trace()
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from astropy import units as u 
from astropy import constants as const
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rc
rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)
font = {'family' : 'normal',
        'weight' : 'light'}
rc('font', **font)

import rheology as rh


params_dict = {
        "mass1": 1.0 * u.Msun,
        "mass2": 1.0 * u.Msun,
        "radius1": 1.0 * u.Rsun,
        "radius2": 1.0 * u.Rsun,
        "rad_gyr1": 2/5 * u.dimensionless_unscaled,
        "rad_gyr2": 2/5 * u.dimensionless_unscaled,
        "tidal_tau": 0.1 * u.s,
        "tidal_Q": 1e6 * u.dimensionless_unscaled,
        "Gconst": const.G,
        "kf": 3/2 * u.dimensionless_unscaled,
        "mean_motion_init": 2*jnp.pi / (10 * u.day),
}

states_dict = {
        "ecc": 0.3 * u.dimensionless_unscaled,
        "omega1": 2*jnp.pi / (1 * u.day),
        "omega2": 2*jnp.pi / (.5 * u.day),
}

times = jnp.logspace(0, 10, 100) * u.yr
ts = jnp.array(times.to(u.s).value)

tm = rh.TidalModelAveragedPlanarCTL(params_dict, states_dict, nterms=5)
output = tm.integrate_evolution(ts)

final = tm.integrate_solution(ts)

# breakpoint()

# # =========================================
plot_vars = ["ecc", "orbital_period", "rotation_period1", "rotation_period2", "semi_major_axis"]
plot_dict = {key: output[key] for key in plot_vars}

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax[0].plot(times.to(u.yr), plot_dict["ecc"], color="black")
ax[0].set_ylabel("eccentricity", fontsize=20)
ax[0].set_xscale("log")

ax[1].plot(times.to(u.yr), plot_dict["orbital_period"], label=r"P$_{\rm orb}$", color="black")
ax[1].plot(times.to(u.yr), plot_dict["rotation_period1"], label=r"P$_{\rm rot1}$", color="b", linestyle="--")
ax[1].plot(times.to(u.yr), plot_dict["rotation_period2"], label=r"P$_{\rm rot2}$", color="g", linestyle="--")
ax[1].set_ylabel("period", fontsize=20)
ax[1].legend(loc="best", fontsize=18)
ax[1].set_xscale("log")

ax[-1].set_xlabel("Time [yr]", fontsize=20)
plt.xlim(min(times).value, max(times).value)
plt.tight_layout()
plt.savefig("integration.png", bbox_inches="tight")
plt.close()
import scipy
from matter_power_spectrum_interpolater import Pk
from generate_mock_galaxies import galaxy_density_chi, redshift_at_chi
import h5py
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

H0=68.1
Omega_0=0.306
c=299792

def window_func(chi):
    return scipy.integrate.quad(
        lambda chi_prime : galaxy_density_chi(chi_prime) * (chi_prime - chi) / (chi_prime * chi),
        chi, 9e3, epsabs = 1e-10, epsrel = 5e-3, limit = 500)[0]

def lps(l):
    # this differs from integral in cosm_setup of fisher_calc_weak_lensing in two ways:
    # calculate gamma^2 instead of lens_pot^2
    # calculate 4 gamma^2, i.e. this is roughly equal to ellipticity squared as defined in add_projected_inertia_tensors.py
    # really what you want is reduced shear, that is a better estimator
    integrand = lambda chi : chi**2 * (1 / (1 + redshift_at_chi(chi)))**(-2) * window_func(chi)**2 * Pk((l+0.5)/chi, redshift_at_chi(chi))[0]
    result =  4 * (l - 1) * l * (l + 1) * (l + 2) * l**(-4) * 9 / 4 * Omega_0**2 * H0**4 * c**(-4) * scipy.integrate.quad(integrand, 1e-5, 9e3, epsabs=1e-30, epsrel=5e-3, limit = 500)[0] # TODO: check 9e3 for chi_last_scatter
    return result

# # --- load the SAME shells the pipeline integrates over ---
# chis_shell = []
# dchi_shell = []
# for i in range(60):
#     with h5py.File(f"data/mass_maps_1024/map_{i}.hdf5", "r") as sf:
#         inner = sf["shell_info"].attrs["comoving_inner_radius"][0]
#         outer = sf["shell_info"].attrs["comoving_outer_radius"][0]
#     chis_shell.append(0.5 * (inner + outer))
#     dchi_shell.append(outer - inner)
# chis_shell = np.array(chis_shell)
# dchi_shell = np.array(dchi_shell)
# z_shell    = np.array([redshift_at_chi(chi) for chi in chis_shell])
# a_shell    = 1.0 / (1.0 + z_shell)

# # precompute the source window at each shell (still a quad over the source n(chi),
# # but evaluated only at the 60 shell centres)
# W_shell = np.array([window_func(chi) for chi in chis_shell])

# def lps_shells(l):
#     # discrete Limber: sum over shells instead of quad over chi
#     k = (l + 0.5) / chis_shell
#     Pk_shell = np.array([Pk(k[s], z_shell[s])[0] for s in range(len(chis_shell))])

#     integrand = chis_shell**2 * a_shell**(-2) * W_shell**2 * Pk_shell
#     integral  = np.sum(integrand * dchi_shell)          # <-- sum(... * dchi), replaces quad

#     pref = (l - 1) * l * (l + 1) * (l + 2) * l**(-4) * 9 / 4 \
#            * Omega_0**2 * H0**4 * c**(-4)
#     return pref * integral

# if __name__ == "__main__":
#     with h5py.File("data/mock_catalogue.hdf5", "r") as data:
#         coords = data["halo_coords_lensed"][:]
#         e1 = data["projected_tensors_lensed"][:, 0]
#         e2 = data["projected_tensors_lensed"][:, 1]

#     nside = 2**9
#     npix = hp.nside2npix(nside)
#     pix = hp.vec2pix(nside, coords[:, 0], coords[:, 1], coords[:, 2])

#     count = np.bincount(pix, minlength=npix)
#     sum_e1 = np.bincount(pix, weights=e1, minlength=npix)
#     sum_e2 = np.bincount(pix, weights=e2, minlength=npix)

#     hit = count > 0
#     e1_map = np.zeros(npix); e1_map[hit] = sum_e1[hit] / count[hit]
#     e2_map = np.zeros(npix); e2_map[hit] = sum_e2[hit] / count[hit]
#     print("std e1:", np.std(e1))
#     print("std e2:", np.std(e2))
#     lmax = 3 * nside - 1
#     alm_E, alm_B = hp.map2alm_spin([e1_map, e2_map], spin=2, lmax=lmax)

#     cl_EE = hp.alm2cl(alm_E)
#     cl_BB = hp.alm2cl(alm_B)

#     var_from_catalogue = np.var(e1) + np.var(e2)
#     ell = np.arange(len(cl_EE))
#     var_from_spectrum  = np.sum((2*ell + 1) / (4*np.pi) * (cl_EE - cl_BB))
#     print(var_from_catalogue, var_from_spectrum)

#     ell = np.arange(len(cl_EE))

#     plt.loglog(ell, cl_EE-cl_BB, label="EE")
#     # plt.loglog(ell, cl_EE - cl_BB, label="EE, shot noise subtracted")
#     # plt.loglog(ell, np.abs(cl_BB), label="|BB|", alpha=0.6)
#     ells_analytical = np.logspace(0, np.log10(ell[-1]), 5)
#     print(ells_analytical)
#     lps_vals = [lps(l) for l in ells_analytical]
#     print(lps_vals)
#     lps_vals_shell = [lps_shells(l) for l in ells_analytical]
#     print(lps_vals_shell)
#     plt.loglog(ells_analytical, lps_vals, linestyle = '--')
#     plt.loglog(ells_analytical, lps_vals_shell, linestyle = '--', label='shelled')
#     plt.title(f"nside={nside}")
#     plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell$"); plt.legend()
#     plt.show()


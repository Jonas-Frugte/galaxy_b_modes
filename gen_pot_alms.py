import healpy as hp
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from tqdm import tqdm
from config import POT_DER_ALMS_L2p8_m9, MASS_MAP_L2p8_m9, OMEGA_M, H0, c, NSIDE_OUTPUT_POT_DER_MAPS

cosmo = FlatLambdaCDM(H0=H0, Om0=OMEGA_M)

def poisson_factor(chi):
    z = z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
    a = 1.0 / (1.0 + z)
    return -1 * (3 / 2) * OMEGA_M * H**2 / c**2 * chi**2 / a

def matter_to_pot_der_alms(matter_map, chi_centr, lmax):

    # get alms of grav pot
    delta_m_map = matter_map / np.mean(matter_map) - 1.0 # TODO: does this actually improve stuff?
    delta_m_alms = hp.map2alm(delta_m_map, lmax=lmax)
    ells = np.arange(lmax + 1)
    # TODO: why not calculate for ell = 0, 1? check later
    delta_m_2_pot_factors = np.zeros(lmax + 1)
    delta_m_2_pot_factors[2:] = poisson_factor(chi_centr) / (ells[2:] * (ells[2:] + 1))
    grav_pot_alms = hp.almxfl(delta_m_alms, delta_m_2_pot_factors)

    # gradient
    grad_alms = hp.almxfl(grav_pot_alms, np.sqrt(ells * (ells+1)))

    # hessian
    kappa_alms = hp.almxfl(grav_pot_alms, -0.5 * ells * (ells + 1))
    pot_to_gamma_E_conversion = np.zeros(lmax + 1)
    pot_to_gamma_E_conversion[2:] = -1 * 0.5 * np.sqrt((ells[2:] - 1) * ells[2:] * (ells[2:] + 1) * (ells[2:] + 2))
    gamma_E_alms = hp.almxfl(grav_pot_alms, pot_to_gamma_E_conversion)

    # flexion
    kappa_to_F = np.zeros(lmax + 1)
    kappa_to_F[2:] = np.sqrt(ells[2:] * (ells[2:] + 1))            # spin 0 -> 1
    gammaE_to_G = np.zeros(lmax + 1)
    gammaE_to_G[2:] = np.sqrt((ells[2:] - 2) * (ells[2:] + 3))     # spin 2 -> 3

    F_alms = hp.almxfl(kappa_alms, kappa_to_F)
    G_alms = hp.almxfl(gamma_E_alms, gammaE_to_G)

    return grad_alms, kappa_alms, gamma_E_alms, F_alms, G_alms

def pot_der_alms_from_FLAMINGO(nside_output=NSIDE_OUTPUT_POT_DER_MAPS):
    lmax = 2 * nside_output

    chis = np.zeros(60)
    grad_alms = np.zeros((60, hp.sphtfunc.Alm.getsize(lmax)), dtype=np.complex64)
    kappa_alms = np.zeros((60, hp.sphtfunc.Alm.getsize(lmax)), dtype=np.complex64)
    gammaE_alms = np.zeros((60, hp.sphtfunc.Alm.getsize(lmax)), dtype=np.complex64)
    F_alms = np.zeros((60, hp.sphtfunc.Alm.getsize(lmax)), dtype=np.complex64)
    G_alms = np.zeros((60, hp.sphtfunc.Alm.getsize(lmax)), dtype=np.complex64)

    for i in tqdm(range(60)):
        # loading mass maps into memory
        # increasing index <-> increasing chi
        # the mass here is actually total amount of mass per pixel. because we work with delta_m instead of mass density directly
        # the conversion factor from mass per pixel to mass per 3D unit area (mass density) cancels out so we can just use it as is
        shell_file = h5py.File(MASS_MAP_L2p8_m9 / f"map_{i}.hdf5", "r")
        mass_map = shell_file["mass_density"][:].astype(np.float32)
        chi_centr = 0.5 * (shell_file["shell_info"].attrs["comoving_inner_radius"][0] + shell_file["shell_info"].attrs["comoving_outer_radius"][0])
        chis[i] = chi_centr
        shell_file.close()

        grad_alms[i], kappa_alms[i], gammaE_alms[i], F_alms[i], G_alms[i] = matter_to_pot_der_alms(mass_map, chi_centr, lmax)

    return grad_alms, kappa_alms, gammaE_alms, F_alms, G_alms

    # np.save(POT_DER_ALMS_L2p8_m9 / f"chis.npy", chis)
    # np.save(POT_DER_ALMS_L2p8_m9 / f"grad_alms.npy", grad_alms)
    # np.save(POT_DER_ALMS_L2p8_m9 / f"kappa_alms.npy", kappa_alms)
    # np.save(POT_DER_ALMS_L2p8_m9 / f"gammaE_alms.npy", gammaE_alms)
    # np.save(POT_DER_ALMS_L2p8_m9 / f"F_alms.npy", F_alms)
    # np.save(POT_DER_ALMS_L2p8_m9 / f"G_alms.npy", G_alms)

def pot_der_alms_from_FLAMINGO_per_shell(sh, nside_output=NSIDE_OUTPUT_POT_DER_MAPS):
    lmax = 2 * nside_output

    shell_file = h5py.File(MASS_MAP_L2p8_m9 / f"map_{sh}.hdf5", "r")
    mass_map = shell_file["mass_density"][:].astype(np.float32)
    chi_centr = 0.5 * (shell_file["shell_info"].attrs["comoving_inner_radius"][0] + shell_file["shell_info"].attrs["comoving_outer_radius"][0])
    shell_file.close()

    return matter_to_pot_der_alms(mass_map, chi_centr, lmax)
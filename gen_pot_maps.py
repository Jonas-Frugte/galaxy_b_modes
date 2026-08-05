import healpy as hp
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from tqdm import tqdm
from config import POT_DER_MAPS_L2p8_m9, MASS_MAP_L2p8_m9, OMEGA_M, H0, c, NSIDE_OUTPUT_POT_DER_MAPS

cosmo = FlatLambdaCDM(H0=H0, Om0=OMEGA_M)

def poisson_factor(chi):
    z = z_at_value(cosmo.comoving_distance, chi * u.Mpc).value
    a = 1.0 / (1.0 + z)
    return -1 * (3 / 2) * OMEGA_M * H**2 / c**2 * chi**2 / a

def kappa_gammaE_to_flexion(kappa_alms, gamma_E_alms, ells, lmax, nside_output):
    """Build the spin-1 (F) and spin-3 (G) flexion maps from convergence and
    shear-E alms, via curved-sky spin-raising.  F = edth kappa, G = edth gamma."""
    kappa_to_F = np.zeros(lmax + 1)
    kappa_to_F[2:] = np.sqrt(ells[2:] * (ells[2:] + 1))            # spin 0 -> 1
    gammaE_to_G = np.zeros(lmax + 1)
    gammaE_to_G[2:] = np.sqrt((ells[2:] - 2) * (ells[2:] + 3))     # spin 2 -> 3

    F_alms = hp.almxfl(kappa_alms, kappa_to_F)
    G_alms = hp.almxfl(gamma_E_alms, gammaE_to_G)
    F1, F2 = hp.alm2map_spin([F_alms, np.zeros_like(F_alms)], nside_output, 1, lmax)
    G1, G2 = hp.alm2map_spin([G_alms, np.zeros_like(G_alms)], nside_output, 3, lmax)
    return F1, F2, G1, G2

def flexion_to_D(F1, F2, G1, G2):
    """Assemble the symmetric third-derivative tensor D_ijk (Bacon eqs 16-17)."""
    D_ttt = -0.5 * (3 * F1 + G1)
    D_ttp = -0.5 * (F2 + G2)
    D_tpp = -0.5 * (F1 - G1)
    D_ppp = -0.5 * (3 * F2 - G2)
    return D_ttt, D_ttp, D_tpp, D_ppp

def matter_to_pot_ders(matter_map, chi_centr, nside_output=None):
    nside_input = hp.npix2nside(len(matter_map))

    if nside_output is None:
        nside_output = nside_input

    lmax = 2 * nside_input # TODO: optimize later

    # get alms of grav pot
    delta_m_map = matter_map / np.mean(matter_map) - 1.0 # TODO: does this actually improve stuff?
    delta_m_alms = hp.map2alm(delta_m_map, lmax=lmax)
    ells = np.arange(lmax + 1)
    # TODO: why not calculate for ell = 0, 1? check later
    delta_m_2_pot_factors = np.zeros(lmax + 1)
    delta_m_2_pot_factors[2:] = poisson_factor(chi_centr) / (ells[2:] * (ells[2:] + 1))
    grav_pot_alms = hp.almxfl(delta_m_alms, delta_m_2_pot_factors)

    # get gradient of grav pot
    pot_map, pot_t_map, pot_p_map = hp.alm2map_der1(grav_pot_alms, nside_output)

    # get hessian of grav pot
    # first get kappa and gamma of grav pot, bc they are easy to convert to in spher harm space
    kappa_alms = hp.almxfl(grav_pot_alms, -0.5 * ells * (ells + 1))
    pot_to_gamma_E_conversion = np.zeros(lmax + 1)
    pot_to_gamma_E_conversion[2:] = -1 * 0.5 * np.sqrt((ells[2:] - 1) * ells[2:] * (ells[2:] + 1) * (ells[2:] + 2))
    gamma_E_alms = hp.almxfl(grav_pot_alms, pot_to_gamma_E_conversion)

    kappa_map = hp.alm2map(kappa_alms, nside_output, lmax=lmax)
    # at this order gamma has no B modes
    gamma1_map, gamma2_map = hp.alm2map_spin([gamma_E_alms, np.zeros_like(gamma_E_alms)], nside_output, 2, lmax)

    # then get hessian of grav pot in terms of kappa and gamma
    pot_tt_map = kappa_map + gamma1_map
    pot_tp_map = gamma2_map
    pot_pp_map = kappa_map - gamma1_map

    # THIRD ORDER DERS
    F1, F2, G1, G2 = kappa_gammaE_to_flexion(kappa_alms, gamma_E_alms, ells, lmax, nside_output)
    pot_ttt_map, pot_ttp_map, pot_tpp_map, pot_ppp_map = flexion_to_D(F1, F2, G1, G2)

    return pot_t_map, pot_p_map, pot_tt_map, pot_tp_map, pot_pp_map, pot_ttt_map, pot_ttp_map, pot_tpp_map, pot_ppp_map, nside_output

def pot_ders_from_FLAMINGO(nside_output=NSIDE_OUTPUT_POT_DER_MAPS):
    chis = np.zeros(60)
    pot_i_maps = np.zeros((60, 2, hp.nside2npix(nside_output)), dtype=np.float32)
    pot_ij_maps = np.zeros((60, 3, hp.nside2npix(nside_output)), dtype=np.float32)
    pot_ijk_maps = np.zeros((60, 4, hp.nside2npix(nside_output)), dtype=np.float32)

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

        pot_i_maps[i, 0], pot_i_maps[i, 1], pot_ij_maps[i, 0], pot_ij_maps[i, 1], pot_ij_maps[i, 2], pot_ijk_maps[i, 0], pot_ijk_maps[i, 1], pot_ijk_maps[i, 2], pot_ijk_maps[i, 3] = matter_to_pot_ders(mass_map, chi_centr, nside_output=nside_output)

    np.save(POT_DER_MAPS_L2p8_m9 / f"chis.npy", chis)
    np.save(POT_DER_MAPS_L2p8_m9 / f"pot_i_maps.npy", pot_i_maps)
    np.save(POT_DER_MAPS_L2p8_m9 / f"pot_ij_maps.npy", pot_ij_maps)
    np.save(POT_DER_MAPS_L2p8_m9 / f"pot_ijk_maps.npy", pot_ijk_maps)

    return chis, pot_i_maps, pot_ij_maps, pot_ijk_maps
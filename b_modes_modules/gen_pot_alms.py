import healpy as hp
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from tqdm import tqdm

from b_modes_modules.filepaths import FilePaths
from b_modes_modules.lens_spec import LensSpec
from b_modes_modules.cosmology_spec import CosmologySpec

def poisson_factor(chi, cosmology: CosmologySpec):
    z = cosmology.z_at_chi(chi)
    a = 1.0 / (1.0 + z)
    return -1 * (3 / 2) * cosmology.omega_m * cosmology.h0**2 / cosmology.c**2 * chi**2 / a

def matter_to_grav_pot_alms(matter_map, chi_centr, lmax, cosmology: CosmologySpec):
    # get alms of grav pot
    delta_m_map = matter_map / np.mean(matter_map) - 1.0 # TODO: does this actually improve stuff?
    delta_m_alms = hp.map2alm(delta_m_map, lmax=lmax)
    ells = np.arange(lmax + 1)
    # TODO: why not calculate for ell = 0, 1? check later
    delta_m_2_pot_factors = np.zeros(lmax + 1)
    delta_m_2_pot_factors[2:] = poisson_factor(chi_centr, cosmology) / (ells[2:] * (ells[2:] + 1))
    return hp.almxfl(delta_m_alms, delta_m_2_pot_factors)

def derived_alms_from_potential(grav_pot_alms, lmax):
    """grad/kappa/gammaE/F/G alms are each just grav_pot_alms times a purely
    ell-dependent factor (no m-mixing), so only grav_pot_alms needs to be
    stored on disk -- these are reconstructed from it on demand, at ~1/5 the
    storage of keeping all five."""
    ells = np.arange(lmax + 1)

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

def grav_pot_alms_from_kappa(kappa_alms):
    """Inverts kappa_alms = almxfl(grav_pot_alms, -0.5*l(l+1)) to recover
    grav_pot_alms. Used to convert old-format stored alms (which kept all
    five derived quantities) into the new grav_pot_alms-only format without
    needing the mass maps or a fresh spherical harmonic transform. The l=0
    factor is zero (grav_pot_alms is zero there by construction anyway, see
    matter_to_grav_pot_alms), so it's left as zero rather than divided."""
    lmax = hp.Alm.getlmax(len(kappa_alms))
    ells = np.arange(lmax + 1)
    inv_factor = np.zeros(lmax + 1)
    inv_factor[2:] = -2.0 / (ells[2:] * (ells[2:] + 1))
    return hp.almxfl(kappa_alms, inv_factor)

def pot_der_alms_from_FLAMINGO_per_shell(sh, lens_spec: LensSpec, filepaths: FilePaths, cosmology: CosmologySpec):
    lmax = 2 * lens_spec.nside_output

    shell_file = h5py.File(filepaths.MASS_MAP / f"map_{sh}.hdf5", "r")
    mass_map = shell_file["total_mass"][:].astype(np.float32)
    chi_centr = 0.5 * (shell_file["shell_info"].attrs["comoving_inner_radius"][0] + shell_file["shell_info"].attrs["comoving_outer_radius"][0])
    shell_file.close()

    return chi_centr, matter_to_grav_pot_alms(mass_map, chi_centr, lmax, cosmology)

def get_shell_chi(sh, filepaths: FilePaths) -> float:
    with h5py.File(filepaths.MASS_MAP / f"map_{sh}.hdf5", "r") as f:
        attrs = f["shell_info"].attrs
        return 0.5 * (attrs["comoving_inner_radius"][0] + attrs["comoving_outer_radius"][0])

def get_stored_alms(sh, filepaths: FilePaths):
    with h5py.File(filepaths.POT_DER_ALMS / filepaths.SHELL_NAME(sh), "r") as f:
        return f["grav_pot_alms"][:]

def process_catalogue(filepaths: FilePaths, lens_spec: LensSpec = LensSpec(), cosmology: CosmologySpec = CosmologySpec()):
    filepaths.POT_DER_ALMS.mkdir(parents=True, exist_ok=True)

    chis = np.zeros(filepaths.NSHELL_MASS_MAPS)
    for sh in tqdm(range(filepaths.NSHELL_MASS_MAPS)):
        out_path = filepaths.POT_DER_ALMS / filepaths.SHELL_NAME(sh)
        if out_path.exists():
            print(f"shell {sh} already done, skipping")
            chis[sh] = get_shell_chi(sh, filepaths)
            continue

        chi_centr, grav_pot_alms = pot_der_alms_from_FLAMINGO_per_shell(
            sh, lens_spec=lens_spec, filepaths=filepaths, cosmology=cosmology)
        chis[sh] = chi_centr

        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with h5py.File(tmp_path, "w") as out:
            out.create_dataset("grav_pot_alms", data=grav_pot_alms.astype(np.complex64))
            out.attrs["shell_index"] = sh
        tmp_path.rename(out_path)
        print(f"shell {sh}: wrote {out_path}")

    np.save(filepaths.CHIS_MASS_MAP, chis)
    print(f"wrote shell chis to {filepaths.CHIS_MASS_MAP}")

    print(f"done, wrote alms for {filepaths.NSHELL_MASS_MAPS} shells to {filepaths.POT_DER_ALMS}")

if __name__ == "__main__":
    process_catalogue(FilePaths())

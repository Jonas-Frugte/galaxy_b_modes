import numpy as np
from pathlib import Path
import numpy as np
import os
import scipy.integrate
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

# pot der maps
NSIDE_INPUT_POT_DER_MAPS = 4096
NSIDE_OUTPUT_POT_DER_MAPS = 4096
NSHELL_MASS_MAPS = 68 # !!! this is different for different sized boxes !!!
NSHELLS_LIGHTCONE = 79
NSIDE_CLS = 512

# Filepaths
# check which system we are on
GEMINI_HOME = Path("/nethome/frugt001")
MAC_HOME = Path("/Users/Frugt001/Desktop")

if GEMINI_HOME.exists():
    HOME = GEMINI_HOME
    DATA = Path("/scratch/frugt001/flamingo/L2p8_m9")
elif MAC_HOME.exists():
    HOME = MAC_HOME
    DATA = Path("/Users/Frugt001/Desktop/flamingo_data/mock_cat_zdist")

REPO     = HOME / "galaxy_b_modes"
PRODUCTS = REPO / "products" # end use products, like graphs or lists that can be directly inspected
CLS = PRODUCTS / "cls"
CLS_UNMASKED = CLS / "unmasked"
CLS_MASKED_SPECIAL = CLS / "masked_special"
CLS_MASKED_RANDOM = CLS / "masked_random"
Z_BINS_PATH = PRODUCTS / "zbins.txt"

# SCRATCH  = Path("/scratch/frugt001")      # bs39 local scratch
# FLAMINGO = SCRATCH / "flamingo"
# FLAMINGO = SCRATCH / "L2p8_m9"
LIGHTCONE = DATA / "lightcones"
MASS_MAP = DATA / f"mass_maps_{NSIDE_INPUT_POT_DER_MAPS}"
CHIS_MASS_MAP = MASS_MAP / f"chis.npy"
SOAP = DATA / "soap"
SHELLS_RESOLVED = DATA / "shells_resolved.hdf5"
POT_DER_MAPS = DATA / f"pot_der_maps_{NSIDE_OUTPUT_POT_DER_MAPS}"
POT_DER_ALMS = DATA / f"pot_der_alms_{NSIDE_OUTPUT_POT_DER_MAPS}"
LENSED_SHELLS = DATA / f"shells_lensed_{NSIDE_INPUT_POT_DER_MAPS}.hdf5"
CONVOLVED_ZS = DATA / f"convolved_zs.hdf5"

def SHELL_NAME(i: int) -> str:
    return f"shell_{i:04d}.hdf5"

# minimum number of particles for a shape tensor to be "well determined"
MIN_NUM_PARTICLES_FOR_SHAPE = 300

# Cosmology
OMEGA_C = 0.27
OMEGA_B = 0.045
OMEGA_M = OMEGA_C + OMEGA_B
H0       = 67
SIGMA8  = 0.83
N_S     = 0.96
c       = 299792.458 # km / s

COSMO = FlatLambdaCDM(H0=H0, Om0=0.306)

# tomography and z stuff
NUM_Z_BINS = 5
def CONVOLVE_ZS(zs: np.ndarray, seed: int = 0) -> np.ndarray:
    gen = np.random.default_rng(seed=seed)
    sigma = 0.05
    zs_convolved = gen.normal(loc=zs, scale=sigma)
    while np.any(zs_convolved < 0.0):
        bad = zs_convolved < 0.0
        zs_convolved[bad] = gen.normal(loc=zs[bad], scale=sigma)
    return zs_convolved
if os.path.exists(Z_BINS_PATH):
    Z_BINS = np.loadtxt(Z_BINS_PATH)
else:
    print("Z_BINS not loaded, path not found")

# which shape tensors to actually lens and do all following calculations with, comes from:
    # fields_soap_tensor = {
    #     "stellar_inertia_tensor": "proj_tensors",
    #     "stellar_inertia_tensor_noniterative": "proj_tensors_nonit",
    #     "stellar_inertia_tensor_reduced": "proj_tensors_red",
    #     "stellar_inertia_tensor_reduced_noniterative": "proj_tensors_red_nonit",
    # }
    # in 'add_projected_inertia_tensors.py'

SHAPE_TYPE_FOR_LENS = "proj_tensors"

z0, beta = 0.64, 1.5
nz_unnorm = lambda z: z**2 * np.exp(-(z / z0)**beta)

# normalize n(z) so it integrates to 1 over z
norm = scipy.integrate.quad(nz_unnorm, 1e-8, 20)[0]
def GAL_DENS_Z(z):
    return nz_unnorm(z) / norm

def Z_AT_CHI(chi):
    return z_at_value(COSMO.comoving_distance, chi * u.Mpc).value

def GAL_DENS_CHI(chi):
    # n(chi) = n(z) |dz/dchi|, transforming the normalized n(z) to a density in chi
    dchi = 0.1
    z = Z_AT_CHI(chi)
    dz_dchi = abs(Z_AT_CHI(chi + dchi) - z) / dchi
    return GAL_DENS_Z(z) * dz_dchi

print(f'''
    CONFIG INFO:\n
    HOME dir: {HOME}
    DATA dir: {DATA}
    NSIDE_INPUT_POT_DER_MAPS = {NSIDE_INPUT_POT_DER_MAPS}
    NSIDE_OUTPUT_POT_DER_MAPS = {NSIDE_OUTPUT_POT_DER_MAPS}
    NSHELL_MASS_MAPS = {NSHELL_MASS_MAPS}
    NSHELLS_LIGHTCONE = {NSHELLS_LIGHTCONE}
    NSIDE_CLS = {NSIDE_CLS}\n
''')
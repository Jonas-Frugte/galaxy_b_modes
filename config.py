import numpy as np
from pathlib import Path
import numpy as np
import os

# pot der maps
NSIDE_INPUT_POT_DER_MAPS = 4096
NSIDE_OUTPUT_POT_DER_MAPS = 4096
NSHELL_MASS_MAPS = 68 # !!! this is different for different sized boxes !!!
NSHELLS_LIGHTCONE = 79
NSIDE_CLS = 512

# Filepaths
HOME     = Path("/nethome/frugt001")
REPO     = HOME / "b-modes"
PRODUCTS = REPO / "products" # end use products, like graphs or lists that can be directly inspected
CLS = PRODUCTS / "cls"
Z_BINS_PATH = PRODUCTS / "zbins.npy"

SCRATCH  = Path("/scratch/frugt001")          # bs39 local scratch
FLAMINGO = SCRATCH / "flamingo"
FLAMINGO_L2p8_m9 = SCRATCH / "L2p8_m9"
LIGHTCONE_L2p8_m9 = FLAMINGO_L2p8_m9 / "lightcones"
MASS_MAP_L2p8_m9 = FLAMINGO_L2p8_m9 / f"mass_maps_{NSIDE_INPUT_POT_DER_MAPS}"
SOAP_L2p8_m9 = FLAMINGO_L2p8_m9 / "soap"
SHELLS_RESOLVED_L2p8_m9 = FLAMINGO_L2p8_m9 / "shells_resolved"
POT_DER_MAPS_L2p8_m9 = FLAMINGO_L2p8_m9 / f"pot_der_maps_{NSIDE_OUTPUT_POT_DER_MAPS}"
POT_DER_ALMS_L2p8_m9 = FLAMINGO_L2p8_m9 / f"pot_der_alms_{NSIDE_OUTPUT_POT_DER_MAPS}"
CHIS_MASS_MAP_L2p8_m9 = MASS_MAP_L2p8_m9 / f"chis.npy"
LENSED_SHELLS_L2p8_m9 = FLAMINGO_L2p8_m9 / f"shells_lensed_{NSIDE_INPUT_POT_DER_MAPS}"
CONVOLVED_ZS_L2p8_m9 = FLAMINGO_L2p8_m9 / f"convolved_zs"

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

# tomography and z stuff
NUM_Z_BINS = 5
def CONVOLVE_ZS(zs: np.ndarray, seed: int = 0) -> np.ndarray:
    gen = np.random.default_rng(seed=seed)
    sigma = 0.05
    zs_convolved = gen.normal(loc=zs, scale=sigma)
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

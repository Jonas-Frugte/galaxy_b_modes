import numpy as np
from pathlib import Path

# pot der maps
NSIDE_INPUT_POT_DER_MAPS = 4096
NSIDE_OUTPUT_POT_DER_MAPS = 4096
NSHELL_MASS_MAPS = 68 # !!! this is different for different sized boxes !!!
NSHELLS_LIGHTCONE = 79

# Filepaths
HOME     = Path("/nethome/frugt001")
REPO     = HOME / "b-modes"
PRODUCTS = REPO / "products" # end use products, like graphs or lists that can be directly inspected
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

# Multipole binning
ELL_MIN, ELL_MAX = 2, 3000
ELL = np.arange(ELL_MIN, ELL_MAX + 1)


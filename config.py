import numpy as np
from pathlib import Path

# Filepaths
HOME     = Path("/nethome/frugt001")
REPO     = HOME / "b-modes"
PRODUCTS = REPO / "products" # end use products, like graphs or lists that can be directly inspected
SCRATCH  = Path("/scratch/frugt001")          # bs39 local scratch
FLAMINGO = SCRATCH / "flamingo"
FLAMINGO_L2p8_m9 = SCRATCH / "L2p8_m9"
LIGHTCONE_L2p8_m9 = FLAMINGO_L2p8_m9 / "lightcones"
MASS_MAP_L2p8_m9_4096 = FLAMINGO_L2p8_m9 / "mass_maps_4096"
SOAP_L2p8_m9 = FLAMINGO_L2p8_m9 / "soap"
SHELLS_RESOLVED_L2p8_m9 = FLAMINGO_L2p8_m9 / "shells_resolved"

# minimum number of particles for a shape tensor to be "well determined"
MIN_NUM_PARTICLES_FOR_SHAPE = 300

# Cosmology
OMEGA_C = 0.27
OMEGA_B = 0.045
H       = 0.67
SIGMA8  = 0.83
N_S     = 0.96

# Multipole binning
ELL_MIN, ELL_MAX = 2, 3000
ELL = np.arange(ELL_MIN, ELL_MAX + 1)
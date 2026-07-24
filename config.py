import numpy as np
from pathlib import Path

# Filepaths
HOME     = Path("/nethome/frugt001")
REPO     = HOME / "b-modes"
SCRATCH  = Path("/scratch/frugt001")          # bs39 local scratch
FLAMINGO = SCRATCH / "flamingo"
FLAMINGO_L2p8_m9 = SCRATCH / "L2p8_m9"
LIGHTCONE_L2p8_m9 = FLAMINGO_L2p8_m9 / "lightcones"
MASS_MAP_L2p8_m9_4096 = FLAMINGO_L2p8_m9 / "mass_maps_4096"
SOAP_L2p8_m9 = FLAMINGO_L2p8_m9 / "soap"

# Cosmology
OMEGA_C = 0.27
OMEGA_B = 0.045
H       = 0.67
SIGMA8  = 0.83
N_S     = 0.96

# Multipole binning
ELL_MIN, ELL_MAX = 2, 3000
ELL = np.arange(ELL_MIN, ELL_MAX + 1)         # computed — YAML can't do this
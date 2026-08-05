import numpy as np
import h5py
from config import MASS_MAP_L2p8_m9, NSHELL_MASS_MAPS

n_shells = NSHELL_MASS_MAPS
chis = np.zeros(n_shells)

for i in range(n_shells):
    with h5py.File(MASS_MAP_L2p8_m9 / f"map_{i}.hdf5", "r") as f:
        inner = f["shell_info"].attrs["comoving_inner_radius"][0]
        outer = f["shell_info"].attrs["comoving_outer_radius"][0]
    chis[i] = 0.5 * (inner + outer)
    print(f"shell {i:3d}: inner={inner:9.2f}  outer={outer:9.2f}  centre={chis[i]:9.2f}")

assert np.all(np.diff(chis) > 0), "shells are not in increasing chi order"

np.savetxt(MASS_MAP_L2p8_m9 / "chis.txt", chis, fmt="%.6f")
np.save(MASS_MAP_L2p8_m9 / "chis.npy", chis)
print(f"\nwrote {n_shells} shell centres to {MASS_MAP_L2p8_m9}")
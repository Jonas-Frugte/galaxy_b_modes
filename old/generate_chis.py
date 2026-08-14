import numpy as np
import h5py
import config

def gen_chi_mass_maps():
    n_shells = config.NSHELL_MASS_MAPS
    chis = np.zeros(n_shells)

    for i in range(n_shells):
        with h5py.File(config.MASS_MAP / f"map_{i}.hdf5", "r") as f:
            inner = f["shell_info"].attrs["comoving_inner_radius"][0]
            outer = f["shell_info"].attrs["comoving_outer_radius"][0]
        chis[i] = 0.5 * (inner + outer)
        print(f"shell {i:3d}: inner={inner:9.2f}  outer={outer:9.2f}  centre={chis[i]:9.2f}")

    assert np.all(np.diff(chis) > 0), "shells are not in increasing chi order"

    np.save(config.CHIS_MASS_MAP, chis)
    print(f"wrote {n_shells} shell centres to {config.CHIS_MASS_MAP}")
    return chis

if __name__ == "__main__":
    gen_chi_mass_maps()

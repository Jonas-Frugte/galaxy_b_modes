'''
Collection of functions to generate new data from existing shells, such as z -> observed z, or scrambled shape tensors
'''

import config
import numpy as np
import h5py

def gen_convolved_zs():
    config.CONVOLVED_ZS.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(config.SHELLS_RESOLVED, "r") as f_in, h5py.File(config.CONVOLVED_ZS, "w") as f_out:
        zs = f_in["redshifts"][:]
        zs_convolved = config.CONVOLVE_ZS(zs)
        f_out.create_dataset("redshifts", data=zs_convolved)
        f_out.create_dataset("track_id", data=f_in["track_id"][:])

def gen_z_bins():
    config.Z_BINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(config.CONVOLVED_ZS, "r") as f:
        zs_all = f["redshifts"][:]

    print(f"{len(zs_all)} galaxies, z range [{zs_all.min():.4f}, {zs_all.max():.4f}]")

    quantiles = np.linspace(0.0, 1.0, config.NUM_Z_BINS + 1)
    edges = np.quantile(zs_all, quantiles)

    # make the outer edges inclusive of everything
    # (this part is slightly autistic but correct)
    edges[0]  = np.nextafter(zs_all.min(), -np.inf)
    edges[-1] = np.nextafter(zs_all.max(),  np.inf)

    counts, _ = np.histogram(zs_all, bins=edges)
    for k in range(config.NUM_Z_BINS):
        print(f"  bin {k}: [{edges[k]:.4f}, {edges[k+1]:.4f})  n = {counts[k]}")

    np.savetxt(config.Z_BINS_PATH, edges, fmt="%.8f")
    return edges

def gen_chi_mass_maps():
    n_shells = config.NSHELL_MASS_MAPS
    chis = np.zeros(n_shells)

    for i in range(n_shells):
        with h5py.File(config.MASS_MAP / f"map_{i}.hdf5", "r") as f:
            inner = f["shell_info"].attrs["comoving_inner_radius"][0]
            outer = f["shell_info"].attrs["comoving_outer_radius"][0]
        chis[i] = 0.5 * (inner + outer)

    assert np.all(np.diff(chis) > 0), "shells are not in increasing chi order"

    np.save(config.CHIS_MASS_MAP, chis)
    print(f"wrote {n_shells} shell centres to {config.CHIS_MASS_MAP}")
    return chis
    
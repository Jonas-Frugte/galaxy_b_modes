'''
Collection of functions to generate new data from existing shells, such as z -> observed z, or scrambled shape tensors
'''

import config
import numpy as np
import h5py

def gen_convolved_zs():
    for i in range(config.NSHELLS_LIGHTCONE):
        path_in = config.SHELLS_RESOLVED_L2p8_m9 / f"shells_{i}.hdf5"
        if not path_in.exists():
            continue
        f_in = h5py.File(path_in)
        f_out = h5py.File(config.CONVOLVED_ZS_L2p8_m9 / f"shells_{i}.hdf5", "w")

        try:
            zs = f_in["redshifts"][:]
            zs_convolved = config.CONVOLVE_ZS(zs)
            f_out.create_dataset("redshifts", data=zs_convolved)
            f_out.create_dataset("track_id", data=f_in["track_id"][:])
        
        finally:
            f_in.close()
            f_out.close()

def gen_z_bins():
    zs_all = []
    for i in range(config.NSHELLS_LIGHTCONE):
        path = config.CONVOLVED_ZS_L2p8_m9 / f"shells_{i}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            zs_all.append(f["redshifts"][:])

    zs_all = np.concatenate(zs_all)
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

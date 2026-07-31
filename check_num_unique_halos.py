import h5py
import config
import numpy as np

shells_resolved_dir = config.SHELLS_RESOLVED_L2p8_m9
out_dir = config.PRODUCTS

num_shells = 79
out_data = np.zeros((79, 5))
for i in range(num_shells):
    with h5py.File(shells_resolved_dir / f"shell_{i:04d}.hdf5", "r") as shell:
        z_lc = shell["redshifts"][:]
        z_lc_min, z_lc_max = z_lc.min(), z_lc.max()
        num_halos = len(z_lc)
        unique, counts = np.unique(shell["SOAP_indexes"][:], return_counts=True)
        num_halos_unique = len(unique)
        avg_replication_num = np.mean(counts)

        out_data[i][:] = np.array([z_lc_min, z_lc_max, num_halos, num_halos_unique, avg_replication_num])
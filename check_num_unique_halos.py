import h5py
import config
import numpy as np

with h5py.File(config.SHELLS_RESOLVED, "r") as shell:
    z_lc = shell["redshifts"][:]
    num_halos = len(z_lc)
    unique, counts = np.unique(shell["SOAP_indexes"][:], return_counts=True)
    num_halos_unique = len(unique)
    avg_replication_num = np.mean(counts)

print(f"z range [{z_lc.min():.4f}, {z_lc.max():.4f}], "
      f"{num_halos} halos, {num_halos_unique} unique, avg replication {avg_replication_num:.2f}")

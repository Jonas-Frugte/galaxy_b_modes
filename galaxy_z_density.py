import h5py
import numpy as np
import matplotlib.pyplot as plt
from generate_mock_galaxies import galaxy_density_z

bins = np.linspace(0.0, 10.0, 401)
np.save("data/L1_m9/gal_z_bins.npy", bins)

try:
    counts = np.load("data/L1_m9/gal_z_bin_number_counts.npy")
except:
    counts = np.zeros(len(bins) - 1)

    for i in range(77):
        with h5py.File(f"data/L1_m9/shells_resolved/shell_{i:04d}.hdf5") as data:
            zs = data["redshifts"][:]
            counts += np.histogram(zs, bins=bins)[0]
            del zs

    np.save("data/L1_m9/gal_z_bin_number_counts.npy", counts)

widths = np.diff(bins)
density = counts / counts.sum() / widths


plt.bar(bins[:-1], density, widths, align="edge")
plt.plot(bins, [galaxy_density_z(z) for z in bins], color = 'black')
plt.title("FLAMINGO redshift gal density vs EUCLID expected sample")
plt.show()
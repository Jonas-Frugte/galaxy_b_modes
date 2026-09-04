import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from b_modes_modules.filepaths import FilePaths

def plot_shell_ps(filepaths: FilePaths = FilePaths(), shells=(0, 20, 40, 60)):
    for sh in shells:
        with h5py.File(filepaths.MASS_MAP / f"map_{sh}.hdf5") as f:
            m = f["total_mass"][:]
        alm = hp.map2alm(m)
        cl = hp.alm2cl(alm)
        plt.loglog(np.arange(len(cl))[2:], cl[2:], label=f"sh={sh}")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_shell_ps()

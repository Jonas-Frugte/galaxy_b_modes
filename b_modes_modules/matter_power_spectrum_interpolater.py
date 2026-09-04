import numpy as np
from scipy.interpolate import RegularGridInterpolator
import glob, os

def load_pk_interpolator(pattern="data/L1_m9/power_spectra/power_matter_*.txt"):
    files = sorted(glob.glob(pattern))
    z_list, pk_list, k_ref = [], [], None
    for fn in files:
        data = np.loadtxt(fn)                 # columns: z, k, P(k)
        z = data[0, 0]
        k = data[:, 1]
        Pk = data[:, 2]
        if k_ref is None:
            k_ref = k
        elif not np.allclose(k, k_ref):
            raise ValueError(f"{fn}: k-grid differs from reference")
        z_list.append(z)
        pk_list.append(Pk)

    z_arr = np.array(z_list)
    order = np.argsort(z_arr)                 # ensure ascending z
    z_arr = z_arr[order]
    logP = np.log(np.array(pk_list)[order])   # (n_z, n_k)
    logk = np.log(k_ref)

    interp = RegularGridInterpolator(
        (z_arr, logk), logP,
        bounds_error=False, fill_value=None)  # None -> linear extrapolation

    def Pk(k, z):
        k = np.atleast_1d(k); z = np.broadcast_to(z, k.shape)  # TODO: what is going on here
        pts = np.column_stack([z, np.log(k)])
        return np.exp(interp(pts))

    return Pk, z_arr, k_ref

Pk, z_grid, k_grid = load_pk_interpolator()
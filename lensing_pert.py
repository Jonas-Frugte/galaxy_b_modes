import healpy as hp
import numpy as np
import h5py
import scipy
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from tqdm import tqdm

Omega_m = 0.306
H0 = 68.1 # km / s / Mpc
c = 299792.458 # km / s

cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

def symm_mat_index_2d(i, j):
    return i + j
        
def symm_mat_index_3d(i, j, k):
    return i + j + k

def simpson_weights(x):
    """Composite Simpson weights for a fixed, possibly non-uniform 1D grid x.
    Returns w such that integral ≈ w @ y for samples y on x.
    Handles even or odd number of intervals (trapezoid on a leftover interval)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = np.zeros(n)
    if n == 1:
        return w
    if n == 2:                      # only one interval -> trapezoid
        h = x[1] - x[0]
        w[:] = [h/2, h/2]
        return w

    # composite Simpson over pairs of intervals [x_{2i}, x_{2i+2}]
    npar = (n - 1) // 2             # number of full Simpson pairs
    for i in range(npar):
        a, b, c = x[2*i], x[2*i+1], x[2*i+2]
        h0, h1 = b - a, c - b
        H = h0 + h1
        # exact quadratic (Simpson) weights for non-uniform sub-intervals
        w[2*i]   += H * (2*h0 - h1) / (6*h0)
        w[2*i+1] += H**3 / (6*h0*h1)
        w[2*i+2] += H * (2*h1 - h0) / (6*h1)
    if (n - 1) % 2 == 1:            # odd leftover interval -> trapezoid
        h = x[-1] - x[-2]
        w[-2] += h/2
        w[-1] += h/2
    return w

def triangular_simpson_matrix(chis):
    n = len(chis)
    M = np.zeros((n, n))
    for a in range(1, n):
        M[a, :a+1] = simpson_weights(chis[:a+1])   # proper Simpson weights up to chi_a
    return M

w_simpson = simpson_weights(chis)
w_simpson_triangular = triangular_simpson_matrix(chis)

def lensing_int(theta, phi, chi_s, order = 1):
    '''
    theta, phi, chi_s : array of floats
    '''
    window_func_weights_delta_angle = np.maximum(1.0 - chis[:, np.newaxis] / chi_s, 0.0)

    # pot_i_maps has shape (nshell, 2, npix)
    ngal = len(chi_s)
    nshell = len(pot_i_maps)

    der_1 = np.array([[
        hp.get_interp_val(pot_i_maps[a, b, :], theta, phi)        
    for b in range(2)] for a in range(nshell)]) / chis[:, np.newaxis, np.newaxis] # nshell * 2 * ngal
    der_2 = np.array([[
        hp.get_interp_val(pot_ij_maps[a, b, :], theta, phi)       
    for b in range(3)] for a in range(nshell)]) / chis[:, np.newaxis, np.newaxis]**2 # nshell * 3 * ngal
    der_3 = np.array([[
        hp.get_interp_val(pot_ijk_maps[a, b, :], theta, phi)       
    for b in range(4)] for a in range(nshell)]) / chis[:, np.newaxis, np.newaxis]**3 # nshell * 4 * ngal
    
    delta_angle_1 = np.zeros((ngal, 2))
    for i in range(2):                        # two transverse components
        integrand = window_func_weights_delta_angle * der_1[:, i, :]      # (nshell * ngal) window * Phi_,i (use that window function is < 0 for chi > chi_s)
        delta_angle_1[:, i] = -2.0 * (w_simpson @ integrand)   # (nshell,) @ (nshell, ngal) -> (ngal,)
        
    psi_ij_1 = np.zeros((ngal, 2, 2))
    for i in range(2):
        for j in range(2):
            integrand = window_func_weights_delta_angle * chis[:, np.newaxis] * der_2[:, symm_mat_index_2d(i, j), :]     # (nshell * ngal) window * chi * Phi_,ij (window function is the same but you get extra chi factor)
            psi_ij_1[:, i, j] = -2.0 * (w_simpson @ integrand)    

    delta_angle_2 = np.zeros((ngal, 2))
    psi_ij_2 = np.zeros((ngal, 2, 2))
    if order >= 2:
        # delta_angle_correction
        for i in range(2):

            integrand_prefactor = np.maximum((1 - chis[:, np.newaxis, np.newaxis] / chi_s), 0.0) * chis[:, np.newaxis, np.newaxis] * np.maximum((1 - chis[np.newaxis, :, np.newaxis] / chis[:, np.newaxis, np.newaxis]), 0.0) # (nshell, nshell, ngal)
            integrand_term1 = der_2[:, np.newaxis, symm_mat_index_2d(i, 0), :] * der_1[np.newaxis, :, 0, :] # (nshell, nshell, ngal)
            integrand_term2 = der_2[:, np.newaxis, symm_mat_index_2d(i, 1), :] * der_1[np.newaxis, :, 1, :]
            integrand = integrand_prefactor * (integrand_term1 + integrand_term2)

            delta_angle_2[:, i] = delta_angle_1 + 4.0 * np.einsum('a,ab,abn->n', w_simpson, w_simpson_triangular, integrand)

        # psi_ij correction
        for i in range(2):
            for j in range(2):

                integrand_prefactor = np.maximum((1 - chis[:, np.newaxis, np.newaxis] / chi_s), 0.0) * chis[:, np.newaxis, np.newaxis] * np.maximum((1 - chis[np.newaxis, :, np.newaxis] / chis[:, np.newaxis, np.newaxis]), 0.0) # (nshell, nshell, ngal)
                integrand_term1 = der_2[:, np.newaxis, symm_mat_index_2d(i, 0), :] * chis[np.newaxis, :, np.newaxis] * der_2[np.newaxis, :, symm_mat_index_2d(0, j), :]
                integrand_term2 = der_2[:, np.newaxis, symm_mat_index_2d(i, 1), :] * chis[np.newaxis, :, np.newaxis] * der_2[np.newaxis, :, symm_mat_index_2d(1, j), :]
                integrand_term3 = chis[:, np.newaxis, np.newaxis] * der_3[:, np.newaxis, symm_mat_index_3d(i, j, 0), :] * der_1[np.newaxis, :, 0, :]
                integrand_term4 = chis[:, np.newaxis, np.newaxis] * der_3[:, np.newaxis, symm_mat_index_3d(i, j, 1), :] * der_1[np.newaxis, :, 1, :]
                integrand = integrand_prefactor * (integrand_term1 + integrand_term2 + integrand_term3 + integrand_term4)

                psi_ij_2[:, i, j] = psi_ij_1 + -4.0 * np.einsum('a,ab,abn->n', w_simpson, w_simpson_triangular, integrand)

    # so if order = 1, then return delta_angle_2 and psi_ij_2 as zero arrays
    return delta_angle_1, psi_ij_1, delta_angle_2, psi_ij_2

def new_coords(theta_old, phi_old, radius_old, delta_angle):
    theta_new = theta_old + delta_angle[:, 0]
    # this is not stable for gals near pole, probably good to eventually change to a rotation matrix method
    phi_new   = phi_old + delta_angle[:, 1] / np.sin(theta_old)   # orthonormal -> coordinate

    # fold theta into [0, pi]; crossing a pole flips phi by pi
    theta_new = theta_new % (2 * np.pi)
    crossed_pole = theta_new > np.pi
    theta_new[crossed_pole] = 2 * np.pi - theta_new[crossed_pole]
    phi_new[crossed_pole] += np.pi
    phi_new = phi_new % (2 * np.pi)

    return hp.ang2vec(theta_new, phi_new) * radius_old[:, np.newaxis]
    
# TODO: check references (seitz @ schneider 1997 and Bartelmann & Schneider 2001) for correctness of this
def new_shape(shapes_old, psi_ij):
    '''
    e = 2g(1+iw)/(1+g^2+w^2), where g = gamma/(1-kappa), w = omega/(1-kappa)
    at leading order we get e = 2g
    '''

    e1 = shapes_old[:, 0]                          # (ngal,)
    e2 = shapes_old[:, 1]
    ngal = len(e1)

    # reconstruct Q up to overall scale (T := 1), batched -> (ngal, 2, 2)
    Q = np.empty((ngal, 2, 2))
    Q[:, 0, 0] = 0.5 * (1.0 + e1)
    Q[:, 0, 1] = 0.5 * e2
    Q[:, 1, 0] = 0.5 * e2
    Q[:, 1, 1] = 0.5 * (1.0 - e1)

    A = np.eye(2)[np.newaxis, :, :] - psi_ij             # broadcasts (2, 2) -> (1, 2, 2) -> (ngal, 2, 2)
    A_inv = np.linalg.inv(A)                       # batched inverse, (ngal, 2, 2)

    # Q_lensed = A_inv @ Q @ A_inv^T, per galaxy
    Q_lensed = A_inv @ Q @ np.transpose(A_inv, (0, 2, 1))   # (ngal, 2, 2)

    T = Q_lensed[:, 0, 0] + Q_lensed[:, 1, 1]      # (ngal,)
    out = np.empty((ngal, 2))
    out[:, 0] = (Q_lensed[:, 0, 0] - Q_lensed[:, 1, 1]) / T
    out[:, 1] = 2.0 * Q_lensed[:, 0, 1] / T
    return out

def lens_catalogue(filenames, batch_size=10000):
    """Apply 1st- and 2nd-order lensing to one or more mock catalogues, in
    galaxy batches to keep the (nshell, nshell, batch) post-Born integrand
    within memory. Writes lensed coords/shapes back into each file."""
    if isinstance(filenames, str):
        filenames = [filenames]

    for filename in filenames:
        print(f"processing {filename}")
        with h5py.File(filename, "a") as gal_shell:
            gal_pos_old = gal_shell["halo_coords"][:]          # (ngal, 3)
            shape_old   = gal_shell["projected_tensors"][:]    # (ngal, 2)
            ngal = len(gal_pos_old)

            radius = np.linalg.norm(gal_pos_old, axis=1)       # (ngal,)
            theta_arr, phi_arr = hp.vec2ang(gal_pos_old)       # each (ngal,)

            # output buffers (filled batch by batch)
            delta_angle_1o = np.zeros((ngal, 2))
            psi_ij_1o      = np.zeros((ngal, 2, 2))
            delta_angle_2o = np.zeros((ngal, 2))
            psi_ij_2o      = np.zeros((ngal, 2, 2))

            for start in tqdm(range(0, ngal, batch_size)):
                sl = slice(start, start + batch_size)
                th, ph, rad = theta_arr[sl], phi_arr[sl], radius[sl]

                delta_angle_1o[sl], psi_ij_1o[sl], delta_angle_2o[sl], psi_ij_2o[sl] = lensing_int(th, ph, rad, order=2)

            # derived lensed quantities (cheap, vectorized over the whole catalogue)
            lensed_coords_1o = new_coords(theta_arr, phi_arr, radius, delta_angle_1o)
            lensed_shapes_1o = new_shape(shape_old, psi_ij_1o)
            lensed_coords_2o = new_coords(theta_arr, phi_arr, radius, delta_angle_2o)
            lensed_shapes_2o = new_shape(shape_old, psi_ij_2o)

            datasets = [
                ("delta_angle",                   delta_angle_1o),
                ("psi_ij",                        psi_ij_1o),
                ("halo_coords_lensed",            lensed_coords_1o),
                ("projected_tensors_lensed",      lensed_shapes_1o),
                ("delta_angle_2o",               delta_angle_2o),
                ("psi_ij_2o",                    psi_ij_2o),
                ("halo_coords_lensed_2o",        lensed_coords_2o),
                ("projected_tensors_lensed_2o",  lensed_shapes_2o),
            ]
            for name, arr in datasets:
                if name in gal_shell:
                    del gal_shell[name]
                gal_shell.create_dataset(name, data=arr.astype(np.float32))

        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9   # GB on macOS
        print(f"  done ({ngal} galaxies, peak RAM {peak:.1f} GB)")


if __name__ == "__main__":

    # lens_catalogue(["data/mock_catalogue_s0.00_1e6gals_z0.5.hdf5", "data/mock_catalogue_s0.00_1e6gals_z1.0.hdf5", "data/mock_catalogue_s0.00_1e6gals_z2.0.hdf5"])
    redshifts = [0.5, 1.0, 2.0]
    rads = [1937.29761134, 3384.000508, 5297.23071338]

    nside = 256
    npix = hp.nside2npix(nside)
    print(npix)
    th, ph = hp.pix2ang(nside, np.arange(npix))

    batch_size = 50000
    for rad, redshift in zip(rads, redshifts):
        omega_map = np.zeros(npix)
        for start in range(0, npix, batch_size):
            sl = slice(start, min(start + batch_size, npix))
            chi_s_batch = np.full(sl.stop - sl.start, rad)
            _, _, _, psi_batch = lensing_int(th[sl], ph[sl], chi_s_batch, order=2)
            omega_map[sl] = 0.5 * (psi_batch[:, 1, 0] - psi_batch[:, 0, 1])
        np.save(f"data/omega_zsource{redshift:.1f}_l400cutoff.npy", omega_map)
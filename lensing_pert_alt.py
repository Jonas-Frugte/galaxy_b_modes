import os
import healpy as hp
import numpy as np
import h5py
from tqdm import tqdm
from ducc0.sht.experimental import synthesis_general

import sht_ders
import gen_pot_alms
import config

def synth(alm, spin, lmax, loc, nthreads=1):
    epsilon=1e-6
    assert alm.shape[-1] == hp.Alm.getsize(lmax), "alm array not packed for this lmax"

    if spin == 0:
        return synthesis_general(
            alm=alm.reshape(1, -1), spin=0, lmax=lmax,
            loc=loc, epsilon=epsilon, nthreads=nthreads)[0]        # single array
    almEB = np.stack([alm, np.zeros_like(alm)])                 # E, B=0
    return synthesis_general(
        alm=almEB, spin=spin, lmax=lmax,
        loc=loc, epsilon=epsilon, nthreads=nthreads)               # (2, npoints)

def truncate_alm(alm, lmax_old, lmax_new):
    out = np.zeros(hp.Alm.getsize(lmax_new), dtype=alm.dtype)
    for m in range(lmax_new + 1):
        n = lmax_new - m + 1
        src = hp.Alm.getidx(lmax_old, m, m)
        dst = hp.Alm.getidx(lmax_new, m, m)
        out[dst:dst + n] = alm[src:src + n]
    return out

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

def lensing_int(theta, phi, chi_s, order=1, alms_from_stored=False, nthreads=os.cpu_count(), lmax_cut=None):
    '''
    theta, phi, chi_s : array of floats, all of length ngal
    '''

    chis = np.load(config.CHIS_MASS_MAP)
    assert np.all((chis[1:] - chis[:-1]) > 0) # chis needs to be increasing

    w_simpson = simpson_weights(chis)
    lmax_full = 2 * config.NSIDE_OUTPUT_POT_DER_MAPS
    lmax = lmax_cut if lmax_cut is not None else lmax_full

    loc = np.array([theta, phi]).T
    # window_func_weights_delta_angle = np.maximum(1.0 - chis[:, np.newaxis] / chi_s, 0.0) # (nshell, ngal)

    # pot_i_maps has shape (nshell, 2, npix)
    ngal = len(chi_s)

    A_int_term = np.zeros((2, ngal))
    B_int_term = np.zeros((2, ngal))
    C_int_term = np.zeros((3, ngal))
    D_int_term = np.zeros((4, ngal))

    A_int_term_prev = np.zeros((2, ngal))
    B_int_term_prev = np.zeros((2, ngal))
    C_int_term_prev = np.zeros((3, ngal))
    D_int_term_prev = np.zeros((4, ngal))
    A=np.zeros((2, ngal))
    B=np.zeros((2, ngal))
    C=np.zeros((3, ngal))
    D=np.zeros((3, ngal))

    delta_angle_1 = np.zeros((ngal, 2))
    psi_ij_1 = np.zeros((ngal, 2, 2))
    delta_angle_2 = np.zeros((ngal, 2))
    psi_ij_2 = np.zeros((ngal, 2, 2))

    for sh in tqdm(range(len(chis))):
        window_func_weights_delta_angle = np.maximum(1.0 - chis[sh] / chi_s, 0.0) # (ngal,)
        if alms_from_stored:
            gradalms, kappaalms, gammaEalms, Falms, Galms = gen_pot_alms.get_stored_alms(sh=sh)
        else:
            gradalms, kappaalms, gammaEalms, Falms, Galms = gen_pot_alms.pot_der_alms_from_FLAMINGO_per_shell(sh=sh)

        if lmax_cut is not None:
            gradalms   = truncate_alm(gradalms, lmax_full, lmax_cut)
            kappaalms  = truncate_alm(kappaalms, lmax_full, lmax_cut)
            gammaEalms = truncate_alm(gammaEalms, lmax_full, lmax_cut)
            Falms      = truncate_alm(Falms, lmax_full, lmax_cut)
            Galms      = truncate_alm(Galms, lmax_full, lmax_cut)

        der_1 = np.array(sht_ders.der_1(*synth(gradalms, spin=1, lmax=lmax, loc=loc, nthreads=nthreads))) / chis[sh] # 2 * ngal

        der_2 = np.array(sht_ders.der_2(synth(kappaalms, spin=0, lmax=lmax, loc=loc, nthreads=nthreads), *synth(gammaEalms, spin=2, lmax=lmax, loc=loc, nthreads=nthreads))) / chis[sh]**2 # 3 * ngal

        der_3 = np.array(sht_ders.flexion_to_D(*synth(Falms, spin=1, lmax=lmax, loc=loc, nthreads=nthreads), *synth(Galms, spin=3, lmax=lmax, loc=loc, nthreads=nthreads))) / chis[sh]**3 # 4 * ngal
        
        
        for i in range(2):                        # two transverse components
            
            integrand = window_func_weights_delta_angle * der_1[i, :]      # (ngal,) window * Phi_,i (use that window function is < 0 for chi > chi_s)
            delta_angle_1[:, i] += -2.0 * (w_simpson[sh] * integrand)   # (1,) * (ngal,) -> (ngal,)
        
        for i in range(2):
            for j in range(2):
                integrand = window_func_weights_delta_angle * chis[sh, np.newaxis] * der_2[symm_mat_index_2d(i, j), :]     # (nshell * ngal) window * chi * Phi_,ij (window function is the same but you get extra chi factor)
                psi_ij_1[:, i, j] += 2.0 * (w_simpson[sh] * integrand)    



        if order >= 2:
            # A = int Phi_,a(chi')
            # B = int chi' Phi_,a(chi')
            # C = int chi' Phi_,aj(chi')
            # D = int chi'^2 Phi_,aj(chi')

            A_int_term = der_1
            B_int_term = der_1 * chis[sh]
            C_int_term = der_2 * chis[sh]
            D_int_term = der_2 * chis[sh]**2
            if sh > 0:
                delta_chi = chis[sh] - chis[sh-1]
                A += (A_int_term_prev + A_int_term) / 2 * delta_chi # in old code would be der_1[sh, a, :]
                B += (B_int_term_prev + B_int_term) / 2 * delta_chi
                C += (C_int_term_prev + C_int_term) / 2 * delta_chi
                D += (D_int_term_prev + D_int_term) / 2 * delta_chi
            A_int_term_prev = A_int_term
            B_int_term_prev = B_int_term
            C_int_term_prev = C_int_term
            D_int_term_prev = D_int_term

            # delta_angle_correction
            for i in range(2):
                integral = np.zeros(ngal)
                for a in range(2):
                    integral += w_simpson[sh] * np.maximum((1 - chis[sh] / chi_s), 0.0) * chis[sh] * (A[a, :] - B[a, :] / chis[sh]) * der_2[symm_mat_index_2d(i, a)]

                delta_angle_2[:, i] += 4.0 * integral

            # psi_ij correction
            for i in range(2):
                for j in range(2):
                    integral = np.zeros(ngal)
                    for a in range(2):
                        # terms 1, 2: Phi_,ia(chi) * chi' * Phi_,aj(chi')
                        integral += w_simpson[sh] * np.maximum((1 - chis[sh] / chi_s), 0.0) * chis[sh] * der_2[symm_mat_index_2d(i, a)] * (C[symm_mat_index_2d(a, j)] - D[symm_mat_index_2d(a, j)] / chis[sh])
                        # terms 3, 4: chi * Phi_,ija(chi) * Phi_,a(chi')
                        integral += w_simpson[sh] * np.maximum((1 - chis[sh] / chi_s), 0.0) * chis[sh] * chis[sh] * der_3[symm_mat_index_3d(i, j, a)] * (A[a, :] - B[a, :] / chis[sh])

                    psi_ij_2[:, i, j] += -4.0 * integral

    if order >= 2:
        delta_angle_2 += delta_angle_1
        psi_ij_2 += psi_ij_1

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

def lens_catalogue(order=2, alms_from_stored=False, nthreads=os.cpu_count(), batch_size=200_000, lmax_cut=None):
    '''
    batch_size gives balance between compute time and ram usage. 
    compute time stays roughly 26 minutes for up to 1e7 gals.
    ram usage for 1e5 gals is 10GB currently, for 2e5 gals its like 11GB ??
    '''


    config.LENSED_SHELLS.parent.mkdir(parents=True, exist_ok=True)
    # if config.LENSED_SHELLS.exists():
    #     print("already lensed, skipping")
    #     return

    with h5py.File(config.SHELLS_RESOLVED, "r") as gal_shell:
        gal_pos_old = gal_shell["halo_coords"][:]
        shape_old   = gal_shell[config.SHAPE_TYPE_FOR_LENS][:]
        track_id    = gal_shell["track_id"][:]
    ngal = len(gal_pos_old)

    radius = np.linalg.norm(gal_pos_old, axis=1)
    theta_arr, phi_arr = hp.vec2ang(gal_pos_old)

    delta_angle_1o = np.zeros((ngal, 2), dtype=np.float32)
    psi_ij_1o      = np.zeros((ngal, 2, 2), dtype=np.float32)
    delta_angle_2o = np.zeros((ngal, 2), dtype=np.float32)
    psi_ij_2o      = np.zeros((ngal, 2, 2), dtype=np.float32)
    lensed_coords_1o = np.zeros((ngal, 3), dtype=np.float32)
    lensed_shapes_1o = np.zeros((ngal, 2), dtype=np.float32)
    lensed_coords_2o = np.zeros((ngal, 3), dtype=np.float32)
    lensed_shapes_2o = np.zeros((ngal, 2), dtype=np.float32)

    for start in range(0, ngal, batch_size):
        sl = slice(start, min(start + batch_size, ngal))
        print(f"batch {sl.start}:{sl.stop} / {ngal}")

        da1, pij1, da2, pij2 = lensing_int(
            theta_arr[sl], phi_arr[sl], radius[sl], order=order, alms_from_stored=alms_from_stored, nthreads=nthreads, lmax_cut=lmax_cut)

        delta_angle_1o[sl] = da1
        psi_ij_1o[sl]      = pij1
        delta_angle_2o[sl] = da2
        psi_ij_2o[sl]      = pij2

        lensed_coords_1o[sl] = new_coords(theta_arr[sl], phi_arr[sl], radius[sl], da1)
        lensed_shapes_1o[sl] = new_shape(shape_old[sl], pij1)
        lensed_coords_2o[sl] = new_coords(theta_arr[sl], phi_arr[sl], radius[sl], da2)
        lensed_shapes_2o[sl] = new_shape(shape_old[sl], pij2)

        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(f"  done ({sl.stop - sl.start} galaxies, peak RAM {peak:.1f} GB)")

    datasets = [
        ("delta_angle",                  delta_angle_1o),
        ("psi_ij",                       psi_ij_1o),
        ("halo_coords_lensed",           lensed_coords_1o),
        ("projected_tensors_lensed",     lensed_shapes_1o),
        ("delta_angle_2o",               delta_angle_2o),
        ("psi_ij_2o",                    psi_ij_2o),
        ("halo_coords_lensed_2o",        lensed_coords_2o),
        ("projected_tensors_lensed_2o",  lensed_shapes_2o),
    ]

    tmp_path = config.LENSED_SHELLS.with_name(config.LENSED_SHELLS.name + ".tmp")
    with h5py.File(tmp_path, "w") as out:
        for name, arr in datasets:
            out.create_dataset(name, data=arr)
        out.create_dataset("track_id", data=track_id)
        out.attrs["order"] = order
        out.attrs["nside"] = config.NSIDE_OUTPUT_POT_DER_MAPS
    tmp_path.rename(config.LENSED_SHELLS)

    print(f"wrote {ngal} galaxies to {config.LENSED_SHELLS}")

if __name__ == "__main__":
    lens_catalogue(alms_from_stored=True, lmax_cut=1000, batch_size = 1_000_001)

    # lens_catalogue(["data/mock_catalogue_s0.00_1e6gals_z0.5.hdf5", "data/mock_catalogue_s0.00_1e6gals_z1.0.hdf5", "data/mock_catalogue_s0.00_1e6gals_z2.0.hdf5"])
    # redshifts = [0.5, 1.0, 2.0]
    # rads = [1937.29761134, 3384.000508, 5297.23071338]

    # nside = 256
    # npix = hp.nside2npix(nside)
    # print(npix)
    # th, ph = hp.pix2ang(nside, np.arange(npix))

    # batch_size = 50000
    # for rad, redshift in zip(rads, redshifts):
    #     omega_map = np.zeros(npix)
    #     for start in range(0, npix, batch_size):
    #         sl = slice(start, min(start + batch_size, npix))
    #         chi_s_batch = np.full(sl.stop - sl.start, rad)
    #         _, _, _, psi_batch = lensing_int(th[sl], ph[sl], chi_s_batch, order=2)
    #         omega_map[sl] = 0.5 * (psi_batch[:, 1, 0] - psi_batch[:, 0, 1])
    #     np.save(f"data/omega_zsource{redshift:.1f}_l400cutoff.npy", omega_map)
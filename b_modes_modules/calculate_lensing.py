import os
import healpy as hp
import numpy as np
import h5py
from tqdm import tqdm
from ducc0.sht.experimental import synthesis_general

from b_modes_modules import sht_ders
from b_modes_modules import gen_pot_alms
from b_modes_modules.filepaths import FilePaths
from b_modes_modules.lens_spec import LensSpec
from b_modes_modules.cosmology_spec import CosmologySpec

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

def lensing_int(theta, phi, chi_s, lens_spec: LensSpec, filepaths: FilePaths,
                cosmology: CosmologySpec = CosmologySpec(), *,
                alms_from_stored=False, nthreads=os.cpu_count()):
    '''
    theta, phi, chi_s : array of floats, all of length ngal
    '''

    order = lens_spec.lens_order
    lmax_cut = lens_spec.lmax_cut

    chis = np.load(filepaths.CHIS_MASS_MAP)
    assert np.all((chis[1:] - chis[:-1]) > 0) # chis needs to be increasing

    w_simpson = simpson_weights(chis)
    lmax_full = 2 * lens_spec.nside_output
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
            gradalms, kappaalms, gammaEalms, Falms, Galms = gen_pot_alms.get_stored_alms(sh=sh, filepaths=filepaths)
        else:
            _, (gradalms, kappaalms, gammaEalms, Falms, Galms) = gen_pot_alms.pot_der_alms_from_FLAMINGO_per_shell(
                sh=sh, lens_spec=lens_spec, filepaths=filepaths, cosmology=cosmology)

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


def lens_catalogue(lens_spec: LensSpec, filepaths: FilePaths,
                   cosmology: CosmologySpec = CosmologySpec(), *,
                   alms_from_stored=False, nthreads=os.cpu_count(), batch_size=200_000):
    '''
    batch_size gives balance between compute time and ram usage. 
    compute time stays roughly 26 minutes for up to 1e7 gals.
    ram usage for 1e5 gals is 10GB currently, for 2e5 gals its like 11GB ??
    '''
    out_path = filepaths.LENSED_SHELLS
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepaths.SHELLS_RESOLVED, "r") as gal_shell:
        gal_pos_old = gal_shell["halo_coords"][:]
        track_id    = gal_shell["track_id"][:]
    ngal = len(gal_pos_old)

    radius = np.linalg.norm(gal_pos_old, axis=1)
    theta_arr, phi_arr = hp.vec2ang(gal_pos_old)

    delta_angle_1o = np.zeros((ngal, 2), dtype=np.float32)
    psi_ij_1o      = np.zeros((ngal, 2, 2), dtype=np.float32)
    delta_angle_2o = np.zeros((ngal, 2), dtype=np.float32)
    psi_ij_2o      = np.zeros((ngal, 2, 2), dtype=np.float32)

    for start in range(0, ngal, batch_size):
        sl = slice(start, min(start + batch_size, ngal))
        print(f"batch {sl.start}:{sl.stop} / {ngal}")

        da1, pij1, da2, pij2 = lensing_int(
            theta_arr[sl], 
            phi_arr[sl], 
            radius[sl], 
            lens_spec=lens_spec, 
            filepaths=filepaths,
            cosmology=cosmology, 
            alms_from_stored=alms_from_stored, 
            nthreads=nthreads
        )

        delta_angle_1o[sl] = da1
        psi_ij_1o[sl]      = pij1
        delta_angle_2o[sl] = da2
        psi_ij_2o[sl]      = pij2

        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
        print(f"  done ({sl.stop - sl.start} galaxies, peak RAM {peak:.1f} GB)")

    datasets = [
        ("delta_angle_1o",                  delta_angle_1o),
        ("psi_ij_1o",                       psi_ij_1o),
        ("delta_angle_2o",               delta_angle_2o),
        ("psi_ij_2o",                    psi_ij_2o),
    ]

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with h5py.File(tmp_path, "w") as out:
        for name, arr in datasets:
            out.create_dataset(name, data=arr)
        out.create_dataset("track_id", data=track_id)
        out.attrs["order"] = lens_spec.lens_order
        out.attrs["nside"] = lens_spec.nside_output
        out.attrs["lmax_cut"] = lens_spec.lmax_cut
    tmp_path.rename(out_path)

    print(f"wrote {ngal} galaxies to {out_path}")
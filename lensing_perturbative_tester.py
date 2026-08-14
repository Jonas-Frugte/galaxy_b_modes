import numpy as np
import healpy as hp
import h5py
import pymaster as nmt
import matplotlib.pyplot as plt

import config

def check_ders():
    import lensing_pert as lp
    # check derivatives and hessian of potential

    # ----- analytic test field: Phi(n) = n^T A n  (pure ell=2 for this A) -----
    A = np.array([[1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]])          # Phi = n_x^2

    chi_centr_test = 1500.0                  # Mpc
    test_nside     = 1024                     # ell=2 is fully resolved; keep the test cheap
    lmax           = 2 * test_nside

    # ----- build the analytic potential map (vectorized, no python pixel loop) -----
    npix = hp.nside2npix(test_nside)
    nvec = np.asarray(hp.pix2vec(test_nside, np.arange(npix)))     # (3, npix), unit vectors
    pot_map_test  = np.einsum('ip,ij,jp->p', nvec, A, nvec)        # n^T A n per pixel
    pot_alms_test = hp.map2alm(pot_map_test, lmax=lmax)

    # ----- invert Poisson to get a matter map that regenerates this potential -----
    # matter_to_pot_ders applies  Phi_lm = C/(l(l+1)) * delta_lm ; invert it:
    ells   = np.arange(lmax + 1)
    inv_fl = np.zeros(lmax + 1)
    inv_fl[2:] = ells[2:] * (ells[2:] + 1) / lp.poisson_factor(chi_centr_test)
    delta_map_test  = hp.alm2map(hp.almxfl(pot_alms_test, inv_fl), test_nside, lmax=lmax)
    matter_map_test = 1.0 + delta_map_test    # mean 1  ->  matter/mean - 1 = delta exactly
                                            # (delta is large/negative here; fine, the test is linear)

    # ----- run the pipeline under test -----
    _, pot_th_map, pot_ph_map, pot_thth_map, pot_thph_map, pot_phph_map, nside_out = lp.matter_to_pot_ders(matter_map_test, chi_centr_test, test_nside)

    # ----- analytic gradient & Hessian at the OUTPUT resolution (vectorized, pole-guarded) -----
    npix_out = hp.nside2npix(nside_out)
    m = np.asarray(hp.pix2vec(nside_out, np.arange(npix_out)))     # (3, npix_out)

    sin_t = np.sqrt(1.0 - m[2]**2)
    good  = sin_t > 1e-3                        # pole guard: theta_hat/phi_hat diverge as sin->0
    s = np.where(good, sin_t, 1.0)             # safe denominator; masked pixels excluded below

    theta_hat = np.array([m[2]*m[0]/s, m[2]*m[1]/s, -1*s])   # (3, npix_out)
    phi_hat   = np.array([-m[1]/s, m[0]/s, np.zeros_like(s)])

    twoA  = 2.0 * A
    An    = A @ m                               # (3, npix_out)
    f_out = np.einsum('ip,ij,jp->p', m, A, m)   # Phi at output pixels (full quadratic -> Euler term)

    # gradient:  grad = 2 P A n   (orthonormal theta/phi components)
    grad_th_true = 2.0 * np.einsum('ip,ip->p', theta_hat, An)
    grad_ph_true = 2.0 * np.einsum('ip,ip->p', phi_hat,   An)

    # covariant Hessian:  P (2A) P^T  -  2 Phi * I
    hes_thth_true = np.einsum('ip,ij,jp->p', theta_hat, twoA, theta_hat) - 2.0 * f_out
    hes_thph_true = np.einsum('ip,ij,jp->p', theta_hat, twoA, phi_hat)
    hes_phph_true = np.einsum('ip,ij,jp->p', phi_hat,   twoA, phi_hat)   - 2.0 * f_out

    # ----- compare -----
    def report(name, num, ana):
        err = np.abs(num - ana)[good]
        print(f"{name:9s}  max|err| = {err.max():.2e}   rms = {np.sqrt((err**2).mean()):.2e}")

    print(f"test_nside={test_nside} -> output nside={nside_out}, comparing {good.sum()} pixels\n")
    report("grad_th",  pot_th_map, grad_th_true)
    report("grad_ph",  pot_ph_map,   grad_ph_true)
    report("hes_thth", pot_thth_map,   hes_thth_true)
    report("hes_thph", pot_thph_map,   hes_thph_true)
    report("hes_phph", pot_phph_map,   hes_phph_true)
    report("2 * kappa", pot_thth_map + pot_phph_map, hes_thth_true + hes_phph_true)
    report("2 * gamma_1", pot_thth_map - pot_phph_map, hes_thth_true - hes_phph_true)
    return None

def test_flexion():
    from lensing_pert import kappa_gammaE_to_flexion
    nside = 512
    lmax = 2 * nside
    ells = np.arange(lmax + 1)

    # any smooth scalar field works as kappa; use a random low-ell map so all
    # m-modes are exercised (not just a single m, which can hide handedness bugs)
    np.random.seed(0)
    kappa_alms = hp.synalm(np.ones(lmax + 1), lmax=lmax)
    # gamma_E from the same potential isn't needed for the F-vs-der1 check;
    # only kappa is, since F = grad(kappa)

    # --- pipeline route: F via spin-raising ---
    F1_pipe, F2_pipe, G1, G2 = kappa_gammaE_to_flexion(
        kappa_alms, np.zeros_like(kappa_alms), ells, lmax, nside)

    # --- independent route: F = grad(kappa) via der1 (already validated) ---
    _, F1_der1, F2_der1 = hp.alm2map_der1(kappa_alms, nside)
    # der1 returns (kappa_map, d/dtheta kappa, 1/sin(theta) d/dphi kappa) = (.,F1,F2)

    # mask poles where der1 is noisiest
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    good = np.sin(theta) > 1e-2

    for name, a, b in [("F1", F1_pipe, F1_der1), ("F2", F2_pipe, F2_der1)]:
        # allow an overall constant: fit ratio rather than assume 1
        ratio = np.median(a[good] / b[good])
        resid = a[good] - ratio * b[good]
        print(f"{name}: best-fit ratio = {ratio:.4f}, "
              f"resid rms / signal rms = {np.std(resid)/np.std(b[good]):.2e}")
    return None

def check_ps():
    # check ellipticity powerspectrum

    import matplotlib.pyplot as plt
    from map_creation import create_cl
    from born_lps_analytic import lps

    nside = 2**9
    ells, cl = create_cl(["data/mock_catalogue.hdf5",], nside)
    # ordering (apparantly) is EE, EB, BE, BB. EB and BE are only different if you are correlating different spin 2 quantities
    plt.loglog(ells, cl[0] - cl[3], label = "EE")
    # plt.loglog(ells(), np.abs(cl[1]), label = "|EB|")
    # plt.loglog(ells(), cl[3], label = "BB")

    ells_analytical = np.logspace(np.log10(2), np.log10(ells[-1]), 5)
    lps_vals = [lps(l) for l in ells_analytical]

    plt.loglog(ells_analytical, lps_vals, label = "analytical")

    plt.legend()
    plt.title(f"nside={nside}")
    plt.show()
    return None

def check_vectorization():
    with h5py.File("data/mock_catalogue_s0.00_1e4gals.hdf5", "r") as data:
        delta_angle = data["delta_angle"][:]
        delta_angle_vectorized = data["delta_angle_vectorized"][:]

        print("typical theta, phi size:", np.mean(np.abs(delta_angle[:, 0])), np.mean(np.abs(delta_angle[:, 1])))

        theta_err = delta_angle[:, 0] - delta_angle_vectorized[:, 0]
        print("theta diff, mean and std:", np.mean(theta_err), np.std(theta_err))

        phi_err = delta_angle[:, 1] - delta_angle_vectorized[:, 1]
        print("phi diff, mean and std:", np.mean(phi_err), np.std(phi_err))

        psi_ij = data["psi_ij"][:]
        psi_ij_vectorized = data["psi_ij_vectorized"][:]

        print("typical psi_ij size:", np.mean(np.abs(psi_ij[:, 0, 0])), np.mean(np.abs(psi_ij[:, 0, 1])), np.mean(np.abs(psi_ij[:, 1, 1])))

        for i in range(2):
            for j in range(2):
                err = psi_ij[:, i, j] - psi_ij_vectorized[:, i, j]
                print(f"psi_ij[{i},{j}] diff, mean and std:", np.mean(err), np.std(err))

        # symmetry check: at first order psi_ij must be symmetric
        symm_err = psi_ij_vectorized[:, 0, 1] - psi_ij_vectorized[:, 1, 0]
        print("symmetry check [0,1]-[1,0], mean and std:", np.mean(symm_err), np.std(symm_err))

        # lensed coordinates: (N, 3)
        coords = data["halo_coords_lensed"][:]
        coords_vectorized = data["halo_coords_lensed_vectorized"][:]
        print("typical coord size:", np.mean(np.abs(coords)))
        for k, name in enumerate(["x", "y", "z"]):
            err = coords[:, k] - coords_vectorized[:, k]
            print(f"coord {name} diff, mean and std:", np.mean(err), np.std(err))

        # lensed shapes (projected tensors): (N, 2)
        shapes = data["projected_tensors_lensed"][:]
        shapes_vectorized = data["projected_tensors_lensed_vectorized"][:]
        print("typical shape size:", np.mean(np.abs(shapes[:, 0])), np.mean(np.abs(shapes[:, 1])))
        for k in range(2):
            err = shapes[:, k] - shapes_vectorized[:, k]
            print(f"shape e{k+1} diff, mean and std:", np.mean(err), np.std(err))

    return None

# TODO: check this test myself later
def test_flexion_G():
    nside = 512
    lmax = 2 * nside
    ells = np.arange(lmax + 1)

    np.random.seed(0)
    kappa_alms = hp.synalm(np.ones(lmax + 1), lmax=lmax)

    # gamma_E from the same scalar: kappa = -1/2 l(l+1) Phi, gamma_E = 1/2 sqrt(...) Phi
    # so gamma_E_alms = -sqrt((l-1)(l+2)/(l(l+1))) * kappa_alms  (the kappa<->gammaE relation)
    # simplest: derive gamma_E directly from kappa to keep them consistent
    kappa_to_gammaE = np.zeros(lmax + 1)
    kappa_to_gammaE[2:] = np.sqrt((ells[2:] - 1) * (ells[2:] + 2) /
                                  (ells[2:] * (ells[2:] + 1)))
    gamma_E_alms = hp.almxfl(kappa_alms, kappa_to_gammaE)   # sign convention pinned by F test consistency

    # --- build G alms (spin-3) via the pipeline's raising factor ---
    gammaE_to_G = np.zeros(lmax + 1)
    gammaE_to_G[2:] = np.sqrt((ells[2:] - 2) * (ells[2:] + 3))     # spin 2 -> 3
    G_alms = hp.almxfl(gamma_E_alms, gammaE_to_G)

    # --- LHS: lower G by one spin (3 -> 2):  bar-edth G ---
    # lowering factor on spin-s is sqrt((l+s)(l-s+1)); for s=3 -> sqrt((l+3)(l-2))
    G_lower = np.zeros(lmax + 1)
    G_lower[3:] = np.sqrt((ells[3:] + 3) * (ells[3:] - 2))
    lhs_alms = hp.almxfl(G_alms, G_lower)

    # --- RHS: raise kappa by two spins (0 -> 2):  edth^2 kappa ---
    k_raise2 = np.zeros(lmax + 1)
    k_raise2[2:] = np.sqrt(ells[2:] * (ells[2:] + 1)) * np.sqrt((ells[2:] - 1) * (ells[2:] + 2))
    rhs_alms = hp.almxfl(kappa_alms, k_raise2)

    # both are spin-2; compare as maps
    lhs1, lhs2 = hp.alm2map_spin([lhs_alms, np.zeros_like(lhs_alms)], nside, 2, lmax)
    rhs1, rhs2 = hp.alm2map_spin([rhs_alms, np.zeros_like(rhs_alms)], nside, 2, lmax)

    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    good = np.sin(theta) > 1e-2

    for name, a, b in [("spin2_1", lhs1, rhs1), ("spin2_2", lhs2, rhs2)]:
        ratio = np.median(a[good] / b[good])
        resid = a[good] - ratio * b[good]
        print(f"G-check {name}: best-fit ratio = {ratio:.4f}, "
              f"resid rms / signal rms = {np.std(resid)/np.std(b[good]):.2e}")


from rotation_lps import get_rotation_power

def plot_omega_cl(z_source=1.0):

    plt.figure()

    with h5py.File(config.LENSED_SHELLS, "r") as f:
        psi_ij_2o = f["psi_ij_2o"][:]
    with h5py.File(config.SHELLS_RESOLVED, "r") as f:
        coords = f["halo_coords"][:]
    
    nside = 128
    lmax = 2 * nside

    omega = 0.5 * (psi_ij_2o[:, 1, 0] - psi_ij_2o[:, 0, 1])
    omega_alm = hp.map2alm(omega, lmax=lmax)
    cl = hp.alm2cl(omega_alm)
    ls = np.arange(len(cl))

    plt.loglog(ls, cl,   label=r"$C_\ell^{\omega\omega}$ post-Born (measured)")

    # analytic CAMB post-Born rotation spectrum
    L_th, cl_th = get_rotation_power(z_source=z_source, lmax=4000)
    plt.loglog(L_th, cl_th, "--", label=r"$C_\ell^{\omega\omega}$ CAMB")

    sigma = cl_th * np.sqrt(2.0 / (2 * L_th + 1))
    plt.fill_between(L_th, cl_th - sigma, cl_th + sigma, alpha=0.2, color="C1", label=r"$\pm1\sigma$ sample variance")

    plt.xlabel(r"$\ell$"); plt.ylabel(r"$C_\ell^{\omega\omega}$")
    plt.title(f"nside={nside}, $z_s$={z_source}")
    plt.legend()
    plt.show()

    # interpolate CAMB onto your ells and take the ratio in the clean low-ell range
    # from scipy.interpolate import interp1d
    # cl_th_interp = interp1d(L_th, cl_th, bounds_error=False)(ls)
    # ratio = cl[ (ls>=2)&(ls<=100) ] / cl_th_interp[ (ls>=2)&(ls<=100) ]
    # print(ratio)
    # print("mean ratio, up to l=100:", np.nanmean(ratio))


if __name__ == "__main__":
    plot_omega_cl()

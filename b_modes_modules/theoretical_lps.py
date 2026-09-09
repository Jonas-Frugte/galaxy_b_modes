import numpy as np
import camb
from camb import postborn
from camb.sources import SplinedSourceWindow

from b_modes_modules.cosmology_spec import CosmologySpec
from b_modes_modules.cls_spec import ClSpec
from b_modes_modules.dataset import get_binned_zs

# def get_rotation_power(z_source, kmax=100, lmax=5000, non_linear=True, lsamp=None):
#     """Analytic post-Born field-rotation power spectrum C_L^{omega omega}
#     (Pratten & Lewis 2016, lowest Limber), for a single source redshift.

#     Returns (L, C_L) where C_L is the rotation auto-spectrum.
#     Cosmology set to match the FLAMINGO pipeline (edit constants if needed).
#     """
#     H0 = h * 100
#     Omega_m = Omega_b + Omega_cdm
#     ombh2 = Omega_b * h**2
#     omch2 = (Omega_m - Omega_b) * h**2

#     pars = camb.CAMBparams()
#     pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2,
#                        mnu=0.06, num_massive_neutrinos=1, nnu=3.044)
#     pars.InitPower.set_params(ns=n_s, As=2.1e-9)   # placeholder As
#     # rescale to sigma_8 = 0.807:
#     pars.set_matter_power(redshifts=[0.], kmax=100)
#     results = camb.get_results(pars)
#     s8_now = results.get_sigma8()[-1]
#     pars.InitPower.set_params(ns=0.967, As=2.1e-9 * (0.807/s8_now)**2)
#     L, Cl = camb.postborn.get_field_rotation_power(
#         pars, kmax=kmax, lmax=lmax, non_linear=non_linear, z_source=z_source, lsamp=lsamp)

#     return L, Cl

def make_window(zs_in_bin, nbins_hist=50):
    nz, edges = np.histogram(zs_in_bin, bins=nbins_hist, density=True)
    z_centers = 0.5 * (edges[1:] + edges[:-1])
    return SplinedSourceWindow(z=z_centers, W=nz, source_type='lensing')

def get_kappa_power(cls_spec: ClSpec, cosmology_spec: CosmologySpec = None, lmax=2000, non_linear=True):
    """Convergence (lensing) power spectra C_L^{kappa kappa} for the redshift
    distributions of cls_spec.tracer_1 and cls_spec.tracer_2 (as selected by
    cls_spec's binning), on the given cosmology. Returns
    (L, C_L^{11}, C_L^{12}, C_L^{22})."""
    cosmology_spec = cosmology_spec or cls_spec.cosmology

    # tell camb what the cosmology is
    h = cosmology_spec.h0 * 0.01
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=100*h,
        ombh2=cosmology_spec.omega_b * h**2,
        omch2=cosmology_spec.omega_c * h**2,
        mnu=cosmology_spec.mnu
    )
    pars.InitPower.set_params(ns=cosmology_spec.ns, As=cosmology_spec.As)
    pars.NonLinear = camb.model.NonLinear_both if non_linear else camb.model.NonLinear_none

    # specify source windows from each tracer's own binned n(z)
    zs_1 = get_binned_zs(cls_spec, 1)
    zs_2 = get_binned_zs(cls_spec, 2)
    pars.SourceWindows = [make_window(zs_1), make_window(zs_2)]
    pars.SourceTerms.limber_windows = True

    pars.set_for_lmax(lmax)
    results = camb.get_results(pars)
    cls = results.get_source_cls_dict(raw_cl=True)
    clkk_11, clkk_12, clkk_22 = cls['W1xW1'], cls['W1xW2'], cls['W2xW2']

    L = np.arange(clkk_12.size)
    l_conversion = 2 * np.sqrt((L-1) * L * (L+1) * (L+2)) / (L * (L + 1))
    return L, clkk_12 * l_conversion**2 # clkk_11 * l_conversion**2, clkk_22 * l_conversion**2

# def camb_nonlinear_pk(z, ks, non_linear=True):
#     """CAMB (Halofit) matter power P(k) at redshift z, on FLAMINGO cosmology.
#     kh in h/Mpc, returns P in (Mpc/h)^3."""
#     pars = camb.CAMBparams()
#     pars.set_cosmology(H0=h*100, ombh2=Omega_b*h**2,
#                        omch2=Omega_cdm*h**2, mnu=0.06)
#     pars.InitPower.set_params(ns=n_s, As=2.099e-9)   # set As to match sigma8=0.807 (rescale as before)
#     pars.set_matter_power(redshifts=[z], kmax=ks.max()*h*1.1)
#     pars.NonLinear = camb.model.NonLinear_both if non_linear else camb.model.NonLinear_none
#     results = camb.get_results(pars)
#     kh_camb, z_camb, pk = results.get_nonlinear_matter_power_spectrum(
#         hubble_units=False, k_hunit=False)
#     return kh_camb, pk[0]


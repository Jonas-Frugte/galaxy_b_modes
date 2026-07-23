import numpy as np
import camb
from camb import postborn
from camb.sources import GaussianSourceWindow
import yaml

with open("data/flamingo_L1000N1800_hydro_fiducial.yml", "r") as f:
    config = yaml.safe_load(f)

cosmo = config["Cosmology"]
h = cosmo["h"]
Omega_b = cosmo["Omega_b"]
Omega_cdm = cosmo["Omega_cdm"]
n_s = cosmo["n_s"]
s_8 = cosmo["s_8"]

def get_rotation_power(z_source, kmax=100, lmax=5000, non_linear=True, lsamp=None):
    """Analytic post-Born field-rotation power spectrum C_L^{omega omega}
    (Pratten & Lewis 2016, lowest Limber), for a single source redshift.

    Returns (L, C_L) where C_L is the rotation auto-spectrum.
    Cosmology set to match the FLAMINGO pipeline (edit constants if needed).
    """
    H0 = h * 100
    Omega_m = Omega_b + Omega_cdm
    ombh2 = Omega_b * h**2
    omch2 = (Omega_m - Omega_b) * h**2

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100*h, ombh2=ombh2, omch2=omch2,
                       mnu=0.06, num_massive_neutrinos=1, nnu=3.044)
    pars.InitPower.set_params(ns=n_s, As=2.1e-9)   # placeholder As
    # rescale to sigma_8 = 0.807:
    pars.set_matter_power(redshifts=[0.], kmax=100)
    results = camb.get_results(pars)
    s8_now = results.get_sigma8()[-1]
    pars.InitPower.set_params(ns=0.967, As=2.1e-9 * (0.807/s8_now)**2)
    L, Cl = camb.postborn.get_field_rotation_power(
        pars, kmax=kmax, lmax=lmax, non_linear=non_linear, z_source=z_source, lsamp=lsamp)

    return L, Cl

def get_kappa_power(z_source, lmax=2000, sigma=0.02, non_linear=True):
    """Convergence (lensing) power spectrum C_L^{kappa kappa} for a source
    at z_source, on the FLAMINGO cosmology. Returns (L, C_L^kappa)."""
    h = 0.681
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100*h, ombh2=0.0486*h**2,
                       omch2=(0.306-0.0486)*h**2 - 0.0006, mnu=0.06)
    pars.InitPower.set_params(ns=0.967, As=2.1e-9)
    pars.NonLinear = camb.model.NonLinear_both if non_linear else camb.model.NonLinear_none
    pars.SourceWindows = [
        GaussianSourceWindow(redshift=z_source, source_type='lensing', sigma=sigma)
    ]
    pars.set_for_lmax(lmax)
    results = camb.get_results(pars)
    cls = results.get_source_cls_dict(raw_cl=True)
    clkk = cls['W1xW1']                      # convergence auto-spectrum for the window
    L = np.arange(clkk.size)
    l_conversion = 2 * np.sqrt((L-1) * L * (L+1) * (L+2)) / (L * (L + 1))
    return L, clkk * l_conversion**2

def camb_nonlinear_pk(z, ks, non_linear=True):
    """CAMB (Halofit) matter power P(k) at redshift z, on FLAMINGO cosmology.
    kh in h/Mpc, returns P in (Mpc/h)^3."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100, ombh2=Omega_b*h**2,
                       omch2=Omega_cdm*h**2, mnu=0.06)
    pars.InitPower.set_params(ns=n_s, As=2.099e-9)   # set As to match sigma8=0.807 (rescale as before)
    pars.set_matter_power(redshifts=[z], kmax=ks.max()*h*1.1)
    pars.NonLinear = camb.model.NonLinear_both if non_linear else camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh_camb, z_camb, pk = results.get_nonlinear_matter_power_spectrum(
        hubble_units=False, k_hunit=False)
    return kh_camb, pk[0]


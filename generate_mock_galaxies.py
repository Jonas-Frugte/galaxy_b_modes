import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
from typing import Literal
import healpy as hp

import config

rng = np.random.default_rng(42)

def nhat_sample(ngals: int) -> np.ndarray:
    theta = np.arccos(1.0 - 2.0 * rng.random(ngals))     # [0, pi]
    phi   = 2.0 * np.pi * rng.random(ngals)              # [0, 2pi)
    nhat  = np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)])                         # (3, N)
    return nhat

def nhat_sample_uniform(nside: int) -> np.ndarray:
    return np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))

def z_sample(ngals: int, z_delta: float = None) -> np.ndarray:
    if z_delta == None:
        z_grid = np.linspace(1e-6, 5.0, 200000)
        cdf = np.cumsum(config.GAL_DENS_Z(z_grid))
        cdf /= cdf[-1]
        zs = np.interp(rng.random(ngals), cdf, z_grid)
    else:
        zs = np.full(ngals, z_delta)

    chis = config.COSMO.comoving_distance(zs).value          # Mpc
    return zs, chis

def shape_sample(ngals: int, intrinsic_shape_sigma: float = 0.0) -> np.ndarray:
    if intrinsic_shape_sigma > 0.0:
        # generate random angle and magnitude of ellipticity
        phi_e = 2.0 * np.pi * np.random.rand(ngals)
        # for the magnitude the correct distribution is the Rayleigh distribution
        # it is the distribution of the size of a vector with components that are independently normally distributed
        e_mag = np.random.rayleigh(intrinsic_shape_sigma, size=ngals)
        bad = e_mag >= 1.0
        while bad.any():
            e_mag[bad] = rng.rayleigh(intrinsic_shape_sigma, size = bad.sum())
            bad = e_mag >= 1.0
        # use 2 * phi_e bc spin 2
        projected_tensors = np.column_stack([e_mag * np.cos(2 * phi_e), e_mag * np.sin(2 * phi_e)]).astype(np.float32)
        # bc of the resampling the effective distribution of e_i is slightly different, so we check that the std is still close to what we wanted
        print(f"Real standard deviation of e_1, e_2 after resampling: {np.std(projected_tensors[:, 0]):.3f}, {np.std(projected_tensors[:, 1]):.3f}")
    else:
        projected_tensors = np.zeros((ngals, 2))

    return projected_tensors

def save_mock(zs: np.ndarray, chis: np.ndarray, nhats: np.ndarray, shapes: np.ndarray):
    halo_coords = (chis * nhats).T
    ngals = len(zs)
    track_id = np.arange(ngals)

    config.SHELLS_RESOLVED.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(config.SHELLS_RESOLVED, "w") as out:
        out.create_dataset("halo_coords", data=halo_coords)
        out.create_dataset("redshifts", data=zs)
        out.create_dataset("track_id", data=track_id)
        out.create_dataset("proj_tensors", data=shapes)

    print(f"wrote {ngals} galaxies to {config.SHELLS_RESOLVED}, "
          f"z in [{zs.min():.3f}, {zs.max():.3f}], chi in [{chis.min():.1f}, {chis.max():.1f}] Mpc")

if __name__ == "__main__":
    ngals = int(1e6)
    intrinsic_shape_sigma = 0.3    # 0.0 = pure lensing
    nside = None
    z_delta = None

    if nside == None:
        nhats = nhat_sample(ngals)
    elif type(nside) == int:
        nhats = nhat_sample_uniform(nside)
        ngals = nhats.shape[1] # nhats has shape (3, ngals)

    zs, chis = z_sample(ngals, z_delta=z_delta)
    shapes = shape_sample(ngals, intrinsic_shape_sigma)

    save_mock(zs, chis, nhats, shapes)

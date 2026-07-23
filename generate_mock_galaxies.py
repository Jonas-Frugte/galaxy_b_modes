import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

# MUST match the pipeline's cosmology — halo_coords radius is the chi_s that
# lensing_int integrates to, so it has to be the same distance-redshift relation.
cosmo = FlatLambdaCDM(H0=68.1, Om0=0.306)

z0, beta = 0.64, 1.5
nz_unnorm = lambda z: z**2 * np.exp(-(z / z0)**beta)

# normalize n(z) so it integrates to 1 over z
import scipy.integrate
norm = scipy.integrate.quad(nz_unnorm, 1e-8, 20)[0]
def galaxy_density_z(z):
    return nz_unnorm(z) / norm

def redshift_at_chi(chi):
    return z_at_value(cosmo.comoving_distance, chi * u.Mpc).value

def galaxy_density_chi(chi):
    # n(chi) = n(z) |dz/dchi|, transforming the normalized n(z) to a density in chi
    dchi = 0.1
    z = redshift_at_chi(chi)
    dz_dchi = abs(redshift_at_chi(chi + dchi) - z) / dchi
    return galaxy_density_z(z) * dz_dchi

if __name__ == "__main__":
    num_gals = int(1e6)
    intrinsic_shape_sigma = 0.0     # 0.0 = pure lensing

    rng = np.random.default_rng(42)

    # ---- sample redshifts by inverse-CDF (no rejection, no singularities) ----
    # z_grid = np.linspace(1e-6, 5.0, 200000)
    # cdf = np.cumsum(nz_unnorm(z_grid)); cdf /= cdf[-1]
    # redshifts = np.interp(rng.random(num_gals), cdf, z_grid)
    z_source_fixed = 0.5
    redshifts = np.full(num_gals, z_source_fixed)
    chi = cosmo.comoving_distance(redshifts).value          # Mpc
    print(cosmo.comoving_distance([0.5, 1.0, 2.0]).value)

    # ---- uniform positions on the sphere (cos theta uniform, NOT theta uniform) ----
    theta = np.arccos(1.0 - 2.0 * rng.random(num_gals))     # [0, pi]
    phi   = 2.0 * np.pi * rng.random(num_gals)              # [0, 2pi)
    nhat  = np.array([np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)])                         # (3, N)
    halo_coords = (chi * nhat).T                             # (N, 3), |coords| = chi

    # ---- intrinsic shapes (projected_tensors = [e1, e2]) ----
    if intrinsic_shape_sigma > 0.0:
        # generate random angle and magnitude of ellipticity
        phi_e = 2.0 * np.pi * np.random.rand(num_gals)
        # for the magnitude the correct distribution is the Rayleigh distribution
        # it is the distribution of the size of a vector with components that are independently normally distributed
        e_mag = np.random.rayleigh(intrinsic_shape_sigma, size=num_gals)
        bad = e_mag >= 1.0
        while bad.any():
            e_mag[bad] = rng.rayleigh(intrinsic_shape_sigma, size = bad.sum())
            bad = e_mag >= 1.0
        # use 2 * phi_e bc spin 2
        projected_tensors = np.column_stack([e_mag * np.cos(2 * phi_e), e_mag * np.sin(2 * phi_e)]).astype(np.float32)
        # bc of the resampling the effective distribution of e_i is slightly different, so we check that the std is still close to what we wanted
        print(f"Real standard deviation of e_1, e_2 after resampling: {np.std(projected_tensors[:, 0]):.3f}, {np.std(projected_tensors[:, 1]):.3f}")
    else:
        projected_tensors = np.zeros((num_gals, 2))

    # with h5py.File(f"data/mock_catalogue_s{intrinsic_shape_sigma:.2f}_1e{np.log10(num_gals):.0f}gals_z{z_source_fixed:.1f}.hdf5", "w") as out:
    #     out.create_dataset("halo_coords",       data=halo_coords.astype(np.float64))
    #     out.create_dataset("redshifts",         data=redshifts.astype(np.float64))
    #     out.create_dataset("projected_tensors", data=projected_tensors.astype(np.float32))

    # print(f"wrote {num_gals} mock sources; z in [{redshifts.min():.2f},{redshifts.max():.2f}], "
    #     f"chi in [{chi.min():.0f},{chi.max():.0f}] Mpc")
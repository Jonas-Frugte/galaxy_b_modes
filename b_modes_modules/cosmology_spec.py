from dataclasses import dataclass
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

@dataclass(frozen=True)
class CosmologySpec:
    omega_c: float = 0.27
    omega_b: float = 0.045
    h0: float = 67.0
    sigma8: float = 0.83
    n_s: float = 0.96
    c: float = 299792.458 # km / s

    @property
    def omega_m(self) -> float:
        return self.omega_c + self.omega_b

    @property
    def astropy_cosmo(self) -> FlatLambdaCDM:
        return FlatLambdaCDM(H0=self.h0, Om0=self.omega_m)

    def z_at_chi(self, chi):
        return z_at_value(self.astropy_cosmo.comoving_distance, chi * u.Mpc).value

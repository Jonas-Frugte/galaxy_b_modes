from pathlib import Path
from dataclasses import dataclass

# machine detection, runs once at import
GEMINI_HOME = Path("/nethome/frugt001")
MAC_HOME = Path("/Users/Frugt001/Desktop")

if GEMINI_HOME.exists():
    ON_GEMINI = True
    HOME = GEMINI_HOME
    DATA_ROOT = Path("/scratch/frugt001/L2p8_m9")
else:
    ON_GEMINI = False
    HOME = MAC_HOME
    DATA_ROOT = Path("/Users/Frugt001/Desktop/flamingo_data")

REPO = HOME / "galaxy_b_modes"

NSIDE_MASS_MAPS = 4096


@dataclass(frozen=True)
class FilePaths:
    '''Input data. One instance per lightcone / catalogue.'''
    CAT_NAME: str = "real_cat_1"

    NSHELL_MASS_MAPS: int = 68 # !!! different for different sized boxes !!!
    NSHELLS_LIGHTCONE: int = 79

    @property
    def DATA(self) -> Path:
        return DATA_ROOT / self.CAT_NAME

    @property
    def RAW_LIGHTCONE(self) -> Path:
        return self.DATA / "raw"

    @property
    def SOAP(self) -> Path:
        '''Shared across catalogues: same halo catalogue underlies every lightcone.'''
        return DATA_ROOT / "soap"

    @property
    def SHELLS_RESOLVED(self) -> Path:
        return self.DATA / "shells_resolved.hdf5"

    @property
    def MASS_MAP(self) -> Path:
        return self.DATA / f"mass_maps_{NSIDE_MASS_MAPS}"

    @property
    def POT_DER_MAPS(self) -> Path:
        return self.DATA / f"pot_der_maps_{NSIDE_MASS_MAPS}"

    @property
    def POT_DER_ALMS(self) -> Path:
        return self.DATA / f"pot_der_alms_{NSIDE_MASS_MAPS}"

    @property
    def CHIS_MASS_MAP(self) -> Path:
        return self.POT_DER_ALMS / "chis.npy"

    @property
    def LENSED_SHELLS(self) -> Path:
        return self.DATA / f"shells_lensed_{NSIDE_MASS_MAPS}.hdf5"

    @property
    def CONVOLVED_ZS(self) -> Path:
        return self.DATA / "convolved_zs.hdf5"

    @staticmethod
    def SHELL_NAME(i: int) -> str:
        return f"shell_{i:04d}.hdf5"


@dataclass(frozen=True)
class ProductPaths:
    '''Output. Small results tracked in the repo, not the big data.'''
    CAT_NAME: str = "real_cat_1"

    @property
    def PRODUCTS(self) -> Path:
        top = "products_gemini" if ON_GEMINI else "products"
        return REPO / top / self.CAT_NAME

    @property
    def CLS(self) -> Path:
        return self.PRODUCTS / "cls"

    @property
    def SNR(self) -> Path:
        return self.PRODUCTS / "snr"

    @property
    def PLOTS(self) -> Path:
        return self.PRODUCTS / "plots"

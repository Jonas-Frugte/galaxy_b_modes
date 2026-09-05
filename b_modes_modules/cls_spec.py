from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal
import itertools
import json
import numpy as np
import scipy.spatial.transform
from b_modes_modules.filepaths import FilePaths, ProductPaths
from b_modes_modules.lens_spec import LensSpec
from b_modes_modules.cosmology_spec import CosmologySpec

@dataclass(frozen=True)
class Tracer:
    bin_num: int | list[int]
    filepaths: FilePaths = FilePaths()
    field_type: Literal["shape", "density"] = "shape"
    lens_order: Literal[0, 1, 2] = 2 # 0 = unlensed
    IA: Literal["real", "off"] = "real" #, "scrambled_linked", "scrambled_not_linked"] = "none"
    scramble_seed: int = 0

    def __post_init__(self):
        bins = (self.bin_num,) if isinstance(self.bin_num, int) else tuple(self.bin_num)
        object.__setattr__(self, "bin_num", tuple(sorted(bins)))

@dataclass(frozen=True)
class Mask:
    kind: Literal["none", "special", "random"] = "none"
    alpha_deg: float = 2.0
    rotation_seed: int = 0

    special_dirs = np.array([v for v in itertools.product([-1, 0, 1], repeat=3)
                         if v != (0, 0, 0)], dtype=float)

    @property
    def cos_alpha(self):
        return np.cos(np.radians(self.alpha_deg))

    @property
    def rot_mat(self):
        rng = np.random.default_rng(self.rotation_seed)
        theta = np.arccos(1.0 - 2.0 * rng.random())
        phi   = 2.0 * np.pi * rng.random()
        rot_nhat = np.array([np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(theta)])
        rot_angle = rng.random() * 2 * np.pi
        return scipy.spatial.transform.Rotation.from_rotvec(rot_nhat * rot_angle)

    def _in_special_mask(self, locs: np.ndarray[float]) -> np.ndarray[bool]:
        '''
        Assumes locs is of shape (ngal, 3) and special dirs of shape (26, 3).
        Returns TRUE for galaxies that ARE NOT IN the special cones.
        '''
        return np.max(np.einsum("ij,kj->ik", locs, self.special_dirs) / np.linalg.norm(locs, axis=1)[:, np.newaxis] / np.linalg.norm(self.special_dirs, axis=1)[np.newaxis, :], axis=1) < self.cos_alpha

    def _rotate(self, locs: np.ndarray[float]) -> np.ndarray[float]:
        return self.rot_mat.apply(locs) # rot mat is not an actual matrix, but a "rotation object"

    def in_mask(self, locs: np.ndarray[float]) -> np.ndarray[bool]:
        if self.kind == "none":
            return np.array([True for _ in range(len(locs))])
        if self.kind == "special":
            return self._in_special_mask(locs)
        if self.kind == "random":
            return self._in_special_mask(self._rotate(locs))


@dataclass(frozen=True)
class ClSpec:
    tracer_1: Tracer
    tracer_2: Tracer
    mask: Mask = field(default_factory=Mask)
    num_z_bins: int = 6
    nside: int = 512
    nlb: int = 32
    lens: LensSpec = field(default_factory=LensSpec)
    cosmology: CosmologySpec = field(default_factory=CosmologySpec)
    products: ProductPaths = field(default_factory=ProductPaths)
    z_bin_edges: tuple[float, ...] | None = None

    sigma_z: float = 0.05
    f_sky: float = 0.36
    n_gal_per_arcmin2: float = 30.0
    sigma_e: float = 0.30

    @property
    def lmax(self) -> int:
        return 3 * self.nside - 1

    @property
    def filename(self) -> str:
        fmt = lambda bins: "-".join(str(b) for b in bins)
        lo, hi = sorted([self.tracer_1.bin_num, self.tracer_2.bin_num])
        return f"bin{fmt(lo)}x{fmt(hi)}.txt"
 
    @property
    def tag(self) -> str:
        return (f"ns{self.nside}_nlb{self.nlb}_{self.mask.kind}_{self.lens.tag}_sz{self.sigma_z}")

    @property
    def out_dir(self) -> Path:
        d = self.products.CLS / self.tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_spec(self):
        path = self.out_dir / "spec.json"
        if path.exists():
            return
        d = asdict(self)
        d.pop("tracer_1")
        d.pop("tracer_2")
        path.write_text(json.dumps(d, indent=2))
        print(f"wrote {path}")

    def add_z_err(self, zs: np.ndarray, seed: int = 0) -> np.ndarray:
        gen = np.random.default_rng(seed=seed)
        zs_convolved = gen.normal(loc=zs, scale=self.sigma_z)
        while np.any(zs_convolved < 0.0):
            bad = zs_convolved < 0.0
            zs_convolved[bad] = gen.normal(loc=zs[bad], scale=self.sigma_z)
        return zs_convolved

    def compute_bin_edges(self, zs: np.ndarray) -> np.ndarray:
        quantiles = np.linspace(0.0, 1.0, self.num_z_bins + 1)
        edges = np.quantile(zs, quantiles)
        edges[0] = np.nextafter(zs.min(), -np.inf)
        edges[-1] = np.nextafter(zs.max(), np.inf)
        return edges
    
    def process_zs(self, zs: np.ndarray) -> tuple[np.ndarray, tuple]:
        zs_w_err = self.add_z_err(zs)
        edges = self.compute_bin_edges(zs_w_err)

        return zs_w_err, edges

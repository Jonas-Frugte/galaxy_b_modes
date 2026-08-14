from dataclasses import dataclass, field
from typing import Literal
import config
import numpy as np
import h5py
import healpy as hp
import pymaster as nmt
import itertools
import scipy

@dataclass(frozen=True)
class Tracer:
    bin_num: int
    field_type: Literal["shape", "density"] = "shape"
    lens_order: Literal[0, 1, 2] = 2 # 0 = unlensed
    scramble: Literal["none", "linked", "not_linked"] = "none"
    scramble_seed: int = 0

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


@dataclass
class DataSet:
    zs: list[float] = field(default_factory=list) # ngal
    pos: list[list[float]] = field(default_factory=list) # ngal 3
    shape: list[list[float]] = field(default_factory=list) # ngal 2
    field_type: Literal["shape", "density"] = "shape"

    def convert_to_list(self):
        self.zs = list(self.zs)
        self.pos = list(self.pos)
        self.shape = list(self.shape)

    def convert_to_array(self):
        self.zs = np.array(self.zs)
        self.pos = np.concatenate(self.pos, axis=0)
        self.shape = np.concatenate(self.shape, axis=0)


def select_dataset(
        tracer_1: Tracer,
        tracer_2: Tracer,
        mask: Mask = Mask(),
    ) -> tuple[DataSet, DataSet]:
    
    datasets = (DataSet(), DataSet())
    for tracer_id in range(2):
        tracer = (tracer_1, tracer_2)[tracer_id]
        datasets[tracer_id].field_type = tracer.field_type

        if tracer.field_type == "shape":
            if tracer.scramble == "none":
                if tracer.lens_order == 0:
                    shape_path = config.SHELLS_RESOLVED
                    ds_name = config.SHAPE_TYPE_FOR_LENS
                if tracer.lens_order == 1:
                    shape_path = config.LENSED_SHELLS
                    ds_name = "projected_tensors_lensed"
                if tracer.lens_order == 2:
                    shape_path = config.LENSED_SHELLS
                    ds_name = "projected_tensors_lensed_2o"
            #     if tracer.lens_order == 2:
            #         shapes = f[""][:][in_bin_mask]
            #     if tracer.lens_order == 3:
            #         shapes = f[""][:][in_bin_mask]
            # if tracer.scramble == "linked":
            #     # TODO
            # if tracer.scramble == "not_linked":
            #     # TODO

        with h5py.File(config.SHELLS_RESOLVED, "r") as f_resolved, h5py.File(config.CONVOLVED_ZS, "r") as f_convolved:
            # check which is in bin and mask
            z_bin_min = config.Z_BINS[tracer.bin_num]
            z_bin_max = config.Z_BINS[tracer.bin_num + 1]
            zs_w_err = f_convolved["redshifts"][:]

            pos_all = f_resolved["halo_coords"][:]
            in_bin_mask = (zs_w_err < z_bin_max) & (z_bin_min < zs_w_err) & mask.in_mask(pos_all)

            datasets[tracer_id].zs.append(zs_w_err[in_bin_mask])
            datasets[tracer_id].pos.append(pos_all[in_bin_mask])

            if tracer.field_type == "shape":
                with h5py.File(shape_path, "r") as f_shape:
                    datasets[tracer_id].shape.append(f_shape[ds_name][:][in_bin_mask])

    return datasets
    
def calc_cl(
        filename: str,
        dataset1: DataSet,
        dataset2: DataSet,
        out_dir = config.CLS,
        nside = config.NSIDE_CLS,
        effective_mask: Literal["count", "pure"] = "count"
    ) -> dict[str, np.ndarray]:
    
    npix = hp.nside2npix(nside)
    lmax = 3 * nside - 1 # NaMaster straight up won't let us use anything else

    fields = []
    for ds in [dataset1, dataset2]:
        ds.convert_to_array()

        pix = hp.vec2pix(nside, ds.pos[:, 0], ds.pos[:, 1], ds.pos[:, 2])
        count = np.bincount(pix, minlength=npix)
        if effective_mask == "count":
            mask = count.astype(float)
        # if effective_mask == "pure":
            # TODO
            # vecs = pix2vec
            # mask = mask.in_mask(vecs)

        hit = count > 0

        if ds.field_type == "shape":
            sum_e1 = np.bincount(pix, weights=ds.shape[:, 0], minlength=npix)
            sum_e2 = np.bincount(pix, weights=ds.shape[:, 1], minlength=npix)

            e1_map = np.zeros(npix)
            e2_map = np.zeros(npix)
  
            e1_map[hit] = sum_e1[hit] / count[hit]
            e2_map[hit] = sum_e2[hit] / count[hit]

            field = nmt.NmtField(mask, [e1_map, e2_map], purify_e=False, purify_b=False, beam=hp.pixwin(nside, pol=True)[1])
            fields.append(field)

    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=8)
    w = nmt.NmtWorkspace.from_fields(*fields, b)
    cl = w.decouple_cell(nmt.workspaces.compute_coupled_cell(*fields))
    export_data = np.zeros((np.shape(cl)[0]+1, np.shape(cl)[1]))
    export_data[:np.shape(cl)[0], :] = cl
    export_data[-1, :] = b.get_effective_ells()
    np.savetxt(out_dir / filename, export_data)
    print(f"wrote {out_dir / filename}")

if __name__ == "__main__":
    mask_dirs = {
        "none": config.CLS_UNMASKED,
        "special": config.CLS_MASKED_SPECIAL,
        "random": config.CLS_MASKED_RANDOM,
    }
    for d in mask_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tracers = [Tracer(bin_num=i, field_type="shape", lens_order=2, scramble="none") for i in range(config.NUM_Z_BINS)]

    for mask_kind in ["none", "special", "random"]:
        mask = Mask(kind=mask_kind)
        for t1, t2 in itertools.combinations_with_replacement(tracers, 2):
            print(f"bin {t1.bin_num}x{t2.bin_num}, mask {mask_kind}")
            dataset1, dataset2 = select_dataset(t1, t2, mask=mask)
            calc_cl(f"bin{t1.bin_num}x{t2.bin_num}.txt", dataset1, dataset2, out_dir=mask_dirs[mask_kind])
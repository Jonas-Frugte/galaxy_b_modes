from dataclasses import dataclass, field
from typing import Literal
import config
import numpy as np
import h5py
import healpy as hp
import pymaster as nmt
import os

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

    def _in_special_mask(self, locs: np.ndarray[float]) -> np.ndarray[bool]:
        # TODO
        raise(TypeError("Not written yet"))
    
    def _rotate(self, locs: np.ndarray[float]) -> np.ndarray[float]:
        # TODO
        raise(TypeError("Not written yet"))

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
        lmax: int = config.LMAX_CORR,
    ) -> tuple[DataSet, DataSet]:
    
    datasets = (DataSet(), DataSet())
    for tracer_id in range(2):
        tracer = (tracer_1, tracer_2)[tracer_id]
        datasets[tracer_id].field_type = tracer.field_type
        
        for sh in range(config.NSHELLS_LIGHTCONE):
            if not os.path.exists(config.SHELLS_RESOLVED_L2p8_m9 / f"shell_{sh}.hdf5"):
                continue

            f_resolved = h5py.File(config.SHELLS_RESOLVED_L2p8_m9 / f"shell_{sh}.hdf5")
            f_convolved = h5py.File(config.CONVOLVED_ZS_L2p8_m9 / f"shell_{sh}.hdf5")
            if tracer.field_type == "shape":
                if tracer.scramble == "none":
                    if tracer.lens_order == 0:
                        shape_dir = config.SHELLS_RESOLVED_L2p8_m9
                        ds_name = config.SHAPE_TYPE_FOR_LENS
                    if tracer.lens_order == 1:
                        shape_dir = config.LENSED_SHELLS_L2p8_m9
                        ds_name = "projected_tensors_lensed"
                    if tracer.lens_order == 2:
                        shape_dir = config.LENSED_SHELLS_L2p8_m9
                        ds_name = "projected_tensors_lensed_2o"
                #     if tracer.lens_order == 2:
                #         shapes = f[""][:][in_bin_mask]
                #     if tracer.lens_order == 3:
                #         shapes = f[""][:][in_bin_mask]
                # if tracer.scramble == "linked":
                #     # TODO
                # if tracer.scramble == "not_linked":
                #     # TODO

                f_shape = h5py.File(shape_dir / f"shell_{sh}.hdf5")

            try:
                # check which is in bin and mask
                z_bin_min = config.Z_BINS[tracer.bin_num]
                z_bin_max = config.Z_BINS[tracer.bin_num + 1]
                zs_w_err = f_convolved["redshifts"][:]

                pos_shell = f_resolved["halo_coords"][:]
                in_bin_mask = (zs_w_err < z_bin_max) & (z_bin_min < zs_w_err) & mask.in_mask(pos_shell)

                if not np.any(in_bin_mask): # true if all gals outside bin
                    continue
                
                datasets[tracer_id].zs.append(zs_w_err[in_bin_mask])

                pos = f_resolved["halo_coords"][:][in_bin_mask]
                datasets[tracer_id].pos.append(pos)
                
                if tracer.field_type == "shape":
                    shape = f_shape[ds_name][:][in_bin_mask]
                    datasets[tracer_id].shape.append(shape)

            finally:
                f_resolved.close()
                f_convolved.close()
                if tracer.field_type == "shape":
                    f_shape.close()
    
    return datasets
    
def calc_cl(
        filename: str,
        dataset1: DataSet, 
        dataset2: DataSet, 
        nside = config.NSIDE_CL,
        effective_mask: Literal["count", "pure"] = "count"
    ) -> dict[str, np.ndarray]:
    
    npix = hp.nside2npix(nside)
    lmax = 2 * nside

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

            field = nmt.NmtField(mask, [e1_map, e2_map], purify_e=False, purify_b=False, beam=hp.pixwin(nside, pol=True, lmax=lmax)[1])
            fields.append(field)

    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=8)
    w = nmt.NmtWorkspace.from_fields(*fields, b)
    cl = w.decouple_cell(nmt.workspaces.compute_coupled_cell(*fields))
    export_data = np.zeros((np.shape(cl)[0]+1, np.shape(cl)[1]))
    export_data[:np.shape(cl)[0], :] = cl
    export_data[-1, :] = b.get_effective_ells()
    np.savetxt(config.CLS / filename, export_data)
from dataclasses import dataclass, field, replace
from typing import Literal
import numpy as np
import h5py


from b_modes_modules.cls_spec import ClSpec
from b_modes_modules.calculate_gal_shape_pos import get_gal_shape_pos

@dataclass
class DataSet:
    zs: list[float] = field(default_factory=list) # ngal
    pos: list[list[float]] = field(default_factory=list) # ngal 3
    shape: list[list[float]] = field(default_factory=list) # ngal 2
    field_type: Literal["shape", "density"] = "shape"
    bin_num: int = -1

    def convert_to_list(self):
        self.zs = list(self.zs)
        self.pos = list(self.pos)
        self.shape = list(self.shape)

    def convert_to_array(self):
        self.zs = np.array(self.zs)
        self.pos = np.concatenate(self.pos, axis=0)
        self.shape = np.concatenate(self.shape, axis=0)

def get_binned_zs(cls_spec: ClSpec, which: Literal[1, 2]) -> np.ndarray:
    """Redshifts (with scatter applied) of galaxies selected by tracer `which`'s bin(s)."""
    tracer = cls_spec.tracer_1 if which == 1 else cls_spec.tracer_2
    with h5py.File(tracer.filepaths.SHELLS_RESOLVED, "r") as f:
        zs = f["redshifts"][:]
    zs_w_err, z_bins = cls_spec.process_zs(zs)
    return zs_w_err[tracer.is_in_bins(zs_w_err, z_bins)]

def get_datasets(cls_spec: ClSpec) -> tuple[DataSet, DataSet, ClSpec]:
    tracer_1, tracer_2 = (cls_spec.tracer_1, cls_spec.tracer_2)

    datasets = (DataSet(), DataSet())
    for tracer_id in range(2):
        tracer = (tracer_1, tracer_2)[tracer_id]
        datasets[tracer_id].field_type = tracer.field_type
        datasets[tracer_id].bin_num = tracer.bin_num

        with h5py.File(tracer.filepaths.SHELLS_RESOLVED, "r") as f:
            zs = f["redshifts"][:]
            pos_all = f["halo_coords"][:]
            if tracer.field_type == "shape" and tracer.lens_order == 0:
                shape_all = f[cls_spec.lens.shape_type][:] if tracer.IA == "real" else np.zeros((len(pos_all), 2))

        # each tracer may point at a different catalogue, so bin on its own zs
        zs_w_err, z_bins = cls_spec.process_zs(zs)
        if tracer_id == 0:
            cls_spec = replace(cls_spec, z_bin_edges=tuple(z_bins))

        if tracer.field_type == "shape" and tracer.lens_order > 0:
            suffix = "1o" if tracer.lens_order == 1 else "2o"
            with h5py.File(tracer.filepaths.LENSED_SHELLS, "r") as f:
                psi_ij = f[f"psi_ij_{suffix}"][:]
                delta_angle = f[f"delta_angle_{suffix}"][:]
            shape_all, pos_all = get_gal_shape_pos(psi_ij, delta_angle, tracer, cls_spec.lens)

        in_bin_mask = tracer.is_in_bins(zs_w_err, z_bins) & cls_spec.mask.in_mask(pos_all)

        datasets[tracer_id].zs.append(zs_w_err[in_bin_mask])
        datasets[tracer_id].pos.append(pos_all[in_bin_mask])

        if tracer.field_type == "shape":
            datasets[tracer_id].shape.append(shape_all[in_bin_mask])

    return datasets[0], datasets[1], cls_spec
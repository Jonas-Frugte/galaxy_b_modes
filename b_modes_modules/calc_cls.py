from dataclasses import dataclass, field, replace
from typing import Literal
import numpy as np
import h5py
import healpy as hp
import pymaster as nmt

from b_modes_modules.cls_spec import ClSpec
from b_modes_modules.dataset import get_datasets
    
def shape_noise_coupled(ds, mask, count, hit, npix):
    """
    Coupled (pseudo-Cl) shape noise, NaMaster eq 37.
    """
    #TODO: pass intrinsic shape of galaxies, not the lensed shapes. decent approximation for now
    sigma_e_sq = np.mean(np.std(ds.shape, axis=0)**2)
    omega_pix = 4 * np.pi / npix
    return omega_pix * np.sum(mask[hit]**2 * sigma_e_sq / count[hit]) / npix

def calc_cl(clspec: ClSpec, store: bool = False) -> dict[str, np.ndarray]:

    # unpack for convenience
    dataset1, dataset2, clspec = get_datasets(clspec)
    nside = clspec.nside

    npix = hp.nside2npix(nside)
    lmax = 3 * nside - 1 # NaMaster straight up won't let us use anything else

    is_auto = dataset1.bin_num == dataset2.bin_num
    # shape noise comes from random intrinsic shapes -- nothing to subtract if IA is off
    apply_noise = is_auto and clspec.tracer_1.IA == "real" and clspec.tracer_2.IA == "real"

    fields = []
    noise = 0.0
    for ds in [dataset1, dataset2]:
        ds.convert_to_array()

        pix = hp.vec2pix(nside, ds.pos[:, 0], ds.pos[:, 1], ds.pos[:, 2])
        count = np.bincount(pix, minlength=npix)
        mask = count.astype(float)
        # TODO: "pure" weighting (analytic mask geometry instead of galaxy count) not implemented

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

            if apply_noise:
                # TODO: this needs to be adjusted for more general maps with varying weights, see namaster eq 37
                noise = shape_noise_coupled(ds, mask, count, hit, npix) 
                cl_noise = np.zeros((4, lmax + 1))
                cl_noise[0] = noise
                cl_noise[3] = noise

    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=clspec.nlb)
    w = nmt.NmtWorkspace.from_fields(*fields, b)
    cl = w.decouple_cell(nmt.workspaces.compute_coupled_cell(*fields), cl_noise = cl_noise if apply_noise else None)


    export_data = np.zeros((np.shape(cl)[0]+1, np.shape(cl)[1]))
    export_data[:np.shape(cl)[0], :] = cl
    export_data[-1, :] = b.get_effective_ells()

    if store:
        # store results
        out_path = clspec.out_dir / clspec.filename
        np.savetxt(out_path, export_data)
        print(f"wrote {out_path}")

        # store spec
        clspec.save_spec()

    return export_data
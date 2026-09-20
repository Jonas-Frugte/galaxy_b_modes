import sys
import shutil
import numpy as np
import pandas as pd
import h5py
import hdfstream

from b_modes_modules.filepaths import FilePaths
from b_modes_modules.lens_spec import LensSpec
from b_modes_modules.data_download_lightcones import download_lightcone_shell

# mapping from lightcone/SOAP hdf5 files to the resolved-subhalo output file (all per shell)
FIELDS_LIGHTCONE = {
    "halo_coords": "halo_coords",
    "redshifts": "redshifts",
    "SOAP_indexes": "SOAP_indexes",
    "track_id": "track_id",
    "is_central": "is_central"
}
FIELDS_SOAP = {
    "total_mass": "masses",
    "stellar_mass": "stellar_masses",
    "track_id": "SOAP_track_id"
}
FIELDS_SOAP_TENSOR = {
    "stellar_inertia_tensor": "proj_tensors",
    "stellar_inertia_tensor_noniterative": "proj_tensors_nonit",
    "stellar_inertia_tensor_reduced": "proj_tensors_red",
    "stellar_inertia_tensor_reduced_noniterative": "proj_tensors_red_nonit",
}
ALL_FIELDS = list(FIELDS_SOAP_TENSOR.values()) + list(FIELDS_SOAP.values()) + list(FIELDS_LIGHTCONE.values())


def project_tensors(tensors, halo_coords, origin=np.array([0, 0, 0])):
    # tensors: (ngal, 6) packed symmetric inertia tensors; halo_coords: (ngal, 3)
    # vectorised over galaxies -- the per-galaxy python loop this replaced was the
    # dominant cost of the whole resolved-subhalo step (~22 us/galaxy)
    tensors = np.asarray(tensors, dtype=float)
    ngal = len(tensors)

    difference = np.asarray(halo_coords, dtype=float) - origin
    los = difference / np.linalg.norm(difference, axis=1)[:, np.newaxis]

    # rebuild the 3x3 symmetric matrices from the 6 packed components
    # TODO: check the trace of the matrix, how it affects stuff, etc.
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = tensors.T
    S = np.empty((ngal, 3, 3))
    S[:, 0, 0] = Ixx; S[:, 1, 1] = Iyy; S[:, 2, 2] = Izz
    S[:, 0, 1] = S[:, 1, 0] = Ixy
    S[:, 0, 2] = S[:, 2, 0] = Ixz
    S[:, 1, 2] = S[:, 2, 1] = Iyz

    # phi_hat = z_hat x los, written out so we skip building a (ngal, 3) z_hat
    phi_hat = np.zeros((ngal, 3))
    phi_hat[:, 0] = -los[:, 1]
    phi_hat[:, 1] = los[:, 0]
    phi_hat_size = np.linalg.norm(phi_hat, axis=1)

    # sentinel = unresolved subhalo; phi_hat_size == 0 = object on the pole axis, e_phi undefined
    bad = np.all(tensors == 0, axis=1) | (phi_hat_size == 0)
    phi_hat /= np.where(bad, 1.0, phi_hat_size)[:, np.newaxis]
    theta_hat = np.cross(phi_hat, los)

    # theta_hat and phi_hat should be collumn but with numpy convention they are rows here, hence transposed
    projection_matrix_transposed = np.stack([theta_hat, phi_hat], axis=1)                          # (ngal, 2, 3)

    Q = projection_matrix_transposed @ S @ projection_matrix_transposed.transpose(0, 2, 1)         # (ngal, 2, 2) projected tensors

    # TO LINEAR ORDER: e_1 = 2 * gamma_1 / (1 - 2 * kappa), e_2 = 2 * gamma_2 / (1 - 2 * kappa)
    # i.e. ellipticity equals twice the reduced shear
    # (this is for a spherical galaxy)
    T = Q[:, 0, 0] + Q[:, 1, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        e_1 = (Q[:, 0, 0] - Q[:, 1, 1]) / T
        e_2 = 2 * Q[:, 0, 1] / T

    out = np.stack([e_1, e_2], axis=1)
    out[bad] = np.nan
    return out


def resolve_shell(filepaths: FilePaths, i: int, min_num_particles: int, batch_size: int = 20_000_000):
    """Reads one already-downloaded raw lightcone shell + the matching SOAP
    snapshot, resolves it, and returns {field_name: array} for whatever
    galaxies passed the particle-count cut (possibly empty).

    The lightcone's own SOAP_indexes field is not trusted -- for at least one
    box (L1_m8) it turned out to be a stale precomputed index, wrong for the
    majority of halos, while TrackId (the actual persistent identifier on
    both sides) still matches correctly. So the SOAP row for each lightcone
    halo is instead reconstructed here by matching TrackIds directly.

    Processed in batches along the lightcone's halo axis -- some shells have
    several billion halos, and reading a full-shell-sized array (even just
    once, even before masking to the tiny kept fraction) can exceed available
    memory. Only the SOAP-side arrays are read in full up front; those have
    tens of millions of rows at most, never billions."""
    soap_dir = filepaths.SOAP
    shell_path = filepaths.RAW_LIGHTCONE / filepaths.SHELL_NAME(i)

    out_fields = {name: [] for name in ALL_FIELDS}

    with h5py.File(shell_path, "r") as lightcone_shell:
        n_halos = lightcone_shell["track_id"].shape[0]

        if n_halos == 0:
            print(f"  shell {i} empty, skipping")
            return {name: None for name in ALL_FIELDS}

        with h5py.File(soap_dir / f"halos_{i:04d}.hdf5", "r") as soap_data:
            z_snap_soap = float(np.atleast_1d(soap_data["Cosmology"].attrs["Redshift"])[0])
            print(f"  shell {i}: {n_halos:,} halos, SOAP snapshot z={z_snap_soap:.3f}")

            # SOAP-side arrays are small (tens of millions of rows, never
            # billions) -- safe to read in full once, both to build the
            # TrackId lookup and to serve properties for kept rows
            soap_track_id_index = pd.Index(soap_data["track_id"][:])
            n_star_full = soap_data["n_star_particles"][:]

            total_found = 0
            total_kept = 0
            zero_tensor_counts = {value: 0 for value in FIELDS_SOAP_TENSOR.values()}

            for start in range(0, n_halos, batch_size):
                end = min(start + batch_size, n_halos)
                lc_track_id = lightcone_shell["track_id"][start:end]

                # -1 where a lightcone TrackId has no match in this snapshot's SOAP catalogue
                soap_row_idx = soap_track_id_index.get_indexer(lc_track_id)
                found = soap_row_idx != -1
                total_found += int(found.sum())

                n_star_batch = np.zeros(end - start, dtype=n_star_full.dtype)
                n_star_batch[found] = n_star_full[soap_row_idx[found]]
                keep = found & (n_star_batch > min_num_particles)
                n_kept = int(keep.sum())
                total_kept += n_kept
                if n_kept == 0:
                    continue

                soap_row_idx_kept = soap_row_idx[keep]
                # read the batch (bounded size) then mask in numpy, rather than
                # a boolean-indexed h5py read over the whole (huge) dataset
                coords_keep = lightcone_shell["halo_coords"][start:end][keep]

                uniq, inv = np.unique(soap_row_idx_kept, return_inverse=True)

                for key, value in FIELDS_SOAP_TENSOR.items():
                    tensors_for_lightcone_keep = soap_data[key][uniq][inv]
                    zero_tensor_counts[value] += int(np.sum(np.all(tensors_for_lightcone_keep == 0, axis=1)))
                    out_fields[value].append(project_tensors(tensors_for_lightcone_keep, coords_keep))

                for key, value in FIELDS_SOAP.items():
                    out_fields[value].append(soap_data[key][uniq][inv])

                for key, value in FIELDS_LIGHTCONE.items():
                    if key == "SOAP_indexes":
                        # the reconstructed index, not the lightcone's own (possibly stale) one
                        out_fields[value].append(soap_row_idx_kept)
                    elif key == "halo_coords":
                        out_fields[value].append(coords_keep)
                    else:
                        out_fields[value].append(lightcone_shell[key][start:end][keep])

            print(f"    {n_halos - total_found:,} of {n_halos:,} lightcone TrackIds "
                  f"not found in SOAP snapshot {i}")
            print(f"    {total_kept:,} resolved (n_star_particles > {min_num_particles})")
            for value, count in zero_tensor_counts.items():
                if count:
                    print(f"    zero tensors in {value}: {count:,}")

    if total_kept == 0:
        return {name: None for name in ALL_FIELDS}

    return {name: np.concatenate(chunks, axis=0) for name, chunks in out_fields.items()}


def merge_resolved_shells(filepaths: FilePaths):
    """Concatenates every per-shell resolved part into the final
    SHELLS_RESOLVED file, then deletes the (now redundant) parts."""
    parts_dir = filepaths.SHELLS_RESOLVED_PARTS
    part_paths = sorted(parts_dir.glob("shell_*.hdf5"))

    out_fields = {name: [] for name in ALL_FIELDS}
    for part_path in part_paths:
        with h5py.File(part_path, "r") as f:
            for name in ALL_FIELDS:
                if name in f:
                    out_fields[name].append(f[name][:])

    with h5py.File(filepaths.SHELLS_RESOLVED, "w") as out:
        for name, chunks in out_fields.items():
            out.create_dataset(name, data=np.concatenate(chunks, axis=0))

    print(f"  merged {len(part_paths)} shell parts into {filepaths.SHELLS_RESOLVED}")

    shutil.rmtree(parts_dir)
    print(f"  deleted intermediate shell parts: {parts_dir}")


def process_resolved_subhalos(filepaths: FilePaths, lens_spec: LensSpec, lightcone: int):
    """Downloads, resolves, and immediately deletes one raw lightcone shell at
    a time -- for higher-resolution boxes the raw shells are far too large
    (100+ GB each) to keep all of them on disk simultaneously. Each shell's
    small resolved result is written to its own file right away (rather than
    only accumulated in memory), so a crash partway through a lightcone can
    resume from the last completed shell instead of losing everything; the
    per-shell parts are merged into one SHELLS_RESOLVED file at the end."""
    if filepaths.SHELLS_RESOLVED.exists():
        print(f"  {filepaths.SHELLS_RESOLVED} already exists, skipping resolved subhalos")
        return

    lightcone_dir = filepaths.RAW_LIGHTCONE
    min_num_particles = lens_spec.min_particles

    parts_dir = filepaths.SHELLS_RESOLVED_PARTS
    parts_dir.mkdir(parents=True, exist_ok=True)

    root_dir = hdfstream.open("cosma", "/")
    lc_dir = root_dir[f"FLAMINGO/{filepaths.BOX_NAME}/{filepaths.BOX_NAME}/halo_lightcone/lightcone{lightcone}"]

    for i in range(filepaths.NSHELLS_LIGHTCONE):
        part_path = parts_dir / filepaths.SHELL_NAME(i)
        if part_path.exists():
            print(f"  shell {i} already resolved, skipping")
            continue

        shell_path = lightcone_dir / filepaths.SHELL_NAME(i)
        download_lightcone_shell(filepaths, lightcone, i, lc_dir=lc_dir)

        try:
            shell_fields = resolve_shell(filepaths, i, min_num_particles)
        finally:
            # delete the raw shell now, whether or not anything was kept from it --
            # the raw shells are the storage bottleneck, so never leave one behind
            shell_path.unlink(missing_ok=True)
            print(f"  shell {i}: raw file deleted")

        tmp_part_path = part_path.with_name(part_path.name + ".tmp")
        with h5py.File(tmp_part_path, "w") as out:
            for name, array in shell_fields.items():
                if array is not None:
                    out.create_dataset(name, data=array)
        tmp_part_path.rename(part_path)
        print(f"  shell {i}: resolved part written")

    merge_resolved_shells(filepaths)
    print(f"  resolved subhalos done: {filepaths.SHELLS_RESOLVED}")


if __name__ == "__main__":
    LIGHTCONE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    CAT_NAME = sys.argv[2] if len(sys.argv) > 2 else f"real_cat_{LIGHTCONE}"
    process_resolved_subhalos(FilePaths(CAT_NAME=CAT_NAME), LensSpec(), LIGHTCONE)

import sys
import shutil
import numpy as np
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


def resolve_shell(filepaths: FilePaths, i: int, min_num_particles: int):
    """Reads one already-downloaded raw lightcone shell + the matching SOAP
    snapshot, resolves it, and returns {field_name: array} for whatever
    galaxies passed the particle-count cut (possibly empty)."""
    soap_dir = filepaths.SOAP
    shell_path = filepaths.RAW_LIGHTCONE / filepaths.SHELL_NAME(i)

    shell_fields = {name: None for name in ALL_FIELDS}

    with h5py.File(shell_path, "r") as lightcone_shell:
        soap_row_idx_lightcone = lightcone_shell["SOAP_indexes"][:]

        if len(soap_row_idx_lightcone) == 0:
            print(f"  shell {i} empty, skipping")
            return shell_fields

        with h5py.File(soap_dir / f"halos_{i:04d}.hdf5", "r") as soap_data:
            # check redshift of soap_snapshot and redshift of lightcone shell first:
            z_lc = lightcone_shell["redshifts"][:]
            z_lc_min, z_lc_max = z_lc.min(), z_lc.max()
            z_snap_soap = float(np.atleast_1d(soap_data["Cosmology"].attrs["Redshift"])[0])
            print(f"  shell {i}: lightcone z=[{z_lc_min:.3f}, {z_lc_max:.3f}], SOAP snapshot z={z_snap_soap:.3f}")

            n_star_lightcone = soap_data["n_star_particles"][:][soap_row_idx_lightcone]
            keep_lightcone = n_star_lightcone > min_num_particles

            if np.sum(keep_lightcone) == 0:
                return shell_fields

            assert np.all(
                soap_data["track_id"][:][soap_row_idx_lightcone] == lightcone_shell["track_id"][:]
            ), "track_ids don't line up"

            # read then mask in numpy: h5py boolean selection is ~5x slower,
            # and halo_coords gets fully read again in the FIELDS_LIGHTCONE loop below anyway
            coords_keep = lightcone_shell["halo_coords"][:][keep_lightcone]

            uniq, inv = np.unique(soap_row_idx_lightcone[keep_lightcone], return_inverse=True)

            for key, value in FIELDS_SOAP_TENSOR.items():
                tensors_for_lightcone_keep = soap_data[key][uniq][inv]
                num_zero_tensors = np.sum(np.all(tensors_for_lightcone_keep == 0, axis=1))
                print(f"    zero tensors in {key}: {num_zero_tensors}")
                shell_fields[value] = project_tensors(tensors_for_lightcone_keep, coords_keep)

            for key, value in FIELDS_SOAP.items():
                shell_fields[value] = soap_data[key][uniq][inv]

            for key, value in FIELDS_LIGHTCONE.items():
                data_array = lightcone_shell[key][:]
                shell_fields[value] = data_array[keep_lightcone]

    return shell_fields


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

import sys
import numpy as np
import h5py

from b_modes_modules.filepaths import FilePaths
from b_modes_modules.lens_spec import LensSpec

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


def los_vec(halo_coord, origin=np.array([0, 0, 0])):
    halo_coord = np.array(halo_coord)
    difference = halo_coord - origin
    return difference / np.linalg.norm(difference)


def project_tensor(tensor, los_vec):
    # tensor: the (6,) packed symmetric inertia tensor for one subhalo
    if np.all(tensor == 0):                 # sentinel = unresolved subhalo
        return [np.nan, np.nan]

    # rebuild the 3x3 symmetric matrix from the 6 packed components
    # TODO: check the trace of the matrix, how it affects stuff, etc.
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = tensor
    S = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])

    los_vec = los_vec / np.linalg.norm(los_vec)   # ensure unit LOS

    phi_hat = np.cross([0, 0, 1], los_vec)
    phi_hat_size = np.linalg.norm(phi_hat)
    if phi_hat_size == 0:                            # object on the pole axis; e_phi undefined
        return [np.nan, np.nan]
    phi_hat /= phi_hat_size
    theta_hat = np.cross(phi_hat, los_vec)

    # theta_hat and phi_hat should be collumn but with numpy convention they are rows here, hence transposed
    projection_matrix_transposed = np.array([theta_hat, phi_hat])

    Q = projection_matrix_transposed @ S @ projection_matrix_transposed.T                         # 2x2 projected tensor

    # TO LINEAR ORDER: e_1 = 2 * gamma_1 / (1 - 2 * kappa), e_2 = 2 * gamma_2 / (1 - 2 * kappa)
    # i.e. ellipticity equals twice the reduced shear
    # (this is for a spherical galaxy)
    T = Q[0, 0] + Q[1, 1]
    e_1 = (Q[0, 0] - Q[1, 1]) / T
    e_2 = 2 * Q[0, 1] / T

    return [e_1, e_2]


def process_resolved_subhalos(filepaths: FilePaths, lens_spec: LensSpec):
    soap_dir = filepaths.SOAP
    lightcone_dir = filepaths.RAW_LIGHTCONE
    min_num_particles = lens_spec.min_particles

    filepaths.SHELLS_RESOLVED.parent.mkdir(parents=True, exist_ok=True)

    out_fields = {value: [] for value in
                  list(FIELDS_SOAP_TENSOR.values()) + list(FIELDS_SOAP.values()) + list(FIELDS_LIGHTCONE.values())}

    for i in range(filepaths.NSHELLS_LIGHTCONE):
        with h5py.File(lightcone_dir / filepaths.SHELL_NAME(i), "r") as lightcone_shell:
            soap_row_idx_lightcone = lightcone_shell["SOAP_indexes"][:]

            if len(soap_row_idx_lightcone) == 0:
                print(f"  shell {i} empty, skipping")
                continue

            with h5py.File(soap_dir / f"halos_{i:04d}.hdf5", "r") as soap_data:
                # check redshift of soap_snapshot and redshift of lightcone shell first:
                z_lc = lightcone_shell["redshifts"][:]
                z_lc_min, z_lc_max = z_lc.min(), z_lc.max()
                z_snap_soap = float(np.atleast_1d(soap_data["Cosmology"].attrs["Redshift"])[0])
                print(f"  shell {i}: lightcone z=[{z_lc_min:.3f}, {z_lc_max:.3f}], SOAP snapshot z={z_snap_soap:.3f}")

                n_star_lightcone = soap_data["n_star_particles"][:][soap_row_idx_lightcone]
                keep_lightcone = n_star_lightcone > min_num_particles

                if np.sum(keep_lightcone) == 0:
                    continue

                assert np.all(
                    soap_data["track_id"][:][soap_row_idx_lightcone] == lightcone_shell["track_id"][:]
                ), "track_ids don't line up"

                coords_keep = lightcone_shell["halo_coords"][keep_lightcone]

                uniq, inv = np.unique(soap_row_idx_lightcone[keep_lightcone], return_inverse=True)

                for key, value in FIELDS_SOAP_TENSOR.items():
                    tensors_for_lightcone_keep = soap_data[key][uniq][inv]
                    num_zero_tensors = np.sum(np.all(tensors_for_lightcone_keep == 0, axis=1))
                    print(f"    zero tensors in {key}: {num_zero_tensors}")

                    tensors_keep_projected = np.array([
                        project_tensor(tensor, los_vec(c)) for tensor, c in zip(tensors_for_lightcone_keep, coords_keep)
                    ])
                    out_fields[value].append(tensors_keep_projected)

                for key, value in FIELDS_SOAP.items():
                    out_fields[value].append(soap_data[key][uniq][inv])

                for key, value in FIELDS_LIGHTCONE.items():
                    data_array = lightcone_shell[key][:]
                    out_fields[value].append(data_array[keep_lightcone])

    with h5py.File(filepaths.SHELLS_RESOLVED, "w") as out:
        for value, chunks in out_fields.items():
            out.create_dataset(value, data=np.concatenate(chunks, axis=0))

    print(f"  resolved subhalos done: {filepaths.SHELLS_RESOLVED}")


if __name__ == "__main__":
    LIGHTCONE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    CAT_NAME = sys.argv[2] if len(sys.argv) > 2 else f"real_cat_{LIGHTCONE}"
    process_resolved_subhalos(FilePaths(CAT_NAME=CAT_NAME), LensSpec(lightcone=LIGHTCONE))

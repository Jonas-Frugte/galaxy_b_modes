import numpy as np
import h5py

import config

soap_dir = config.SOAP_L2p8_m9
lightcone_dir = config.LIGHTCONE_L2p8_m9
shells_resolved_dir = config.SHELLS_RESOLVED_L2p8_m9
min_num_particles = config.MIN_NUM_PARTICLES_FOR_SHAPE

# create dir if it doesn't already exist
shells_resolved_dir.mkdir(parents=True, exist_ok=True) 

def los_vec(halo_coord, origin = np.array([0, 0, 0])):
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

# mapping from SOAP hdf5 files to projected shape tensor hdf5 file (all per shell)
fields_lightcone = {
    "halo_coords": "halo_coords",
    "redshifts": "redshifts",
    "SOAP_indexes": "SOAP_indexes" 
}
fields_soap = {
    "total_mass": "masses",
    "stellar_mass": "stellar_masses",
    "halo_catalogue_index": "SOAP_index_from_SOAP_cat"
}
fields_soap_tensor = {
    "stellar_inertia_tensor": "proj_tensors",
    "stellar_inertia_tensor_noniterative": "proj_tensors_nonit",
    "stellar_inertia_tensor_reduced": "proj_tensors_red",
    "stellar_inertia_tensor_reduced_noniterative": "proj_tensors_red_nonit",
}
for i in range(79): # shell number
    with h5py.File(lightcone_dir / f"shell_{i:04d}.hdf5", "r") as lightcone_shell:
        soap_idx_lightcone = lightcone_shell["SOAP_indexes"][:]

        if len(soap_idx_lightcone) == 0:
            print(f"Shell {i} empty, skipping")
            continue

        with h5py.File(soap_dir / f"halos_{i:04d}.hdf5", "r") as soap_data:
            soap_idx_soap_data = soap_data["halo_catalogue_index"][:]

            # check redshift of soap_snapshot and redshift of lightcone shell first:
            z_lc = lightcone_shell["redshifts"][:]
            z_lc_min, z_lc_max = z_lc.min(), z_lc.max()
            z_snap_soap = float(np.atleast1d(soap_data["Cosmology"].attrs["Redshift"])[0])
            print(f"i={i}. redshift of lightcone shell: [{z_lc_min, z_lc_max}]. redshift of SOAP snapshot: {z_snap_soap}")

            # the index is the same as the index of the lightcone, the value at each index is the corresponding row in the soap_data, so if you index a row from the soap table you get a new row mapped to the appropriate places to match the lightcone
            if np.all(soap_idx_soap_data[:-1] < soap_idx_soap_data[1:]):
                lightcone_to_soap_data = np.searchsorted(soap_idx_soap_data, soap_idx_lightcone)
            else:
                sorter = np.argsort(soap_idx_soap_data)
                lightcone_to_soap_data = sorter[np.searchsorted(soap_idx_soap_data, soap_idx_lightcone, sorter=sorter)]

            assert lightcone_to_soap_data.max() < soap_idx_soap_data.size, "lightcone ID beyond catalogue"
            assert np.all(soap_idx_soap_data[lightcone_to_soap_data] == soap_idx_lightcone), "unmatched halo"

            n_star_lightcone = soap_data["n_star_particles"][:][lightcone_to_soap_data]
            keep_lightcone = n_star_lightcone > min_num_particles

            coords_keep = lightcone_shell["halo_coords"][keep_lightcone]

            with h5py.File(shells_resolved_dir / f"shell_{i:04d}.hdf5", "w") as out:

                for key, value in fields_soap_tensor.items():
                    tensors = soap_data[key][:]
                    tensors_for_lightcone = tensors[lightcone_to_soap_data]
                    tensors_for_lightcone_keep = tensors_for_lightcone[keep_lightcone]

                    num_zero_tensors = np.sum(np.all(tensors_for_lightcone_keep == 0, axis=1))
                    print(f"Number of zero tensors in {key}: {num_zero_tensors}")

                    tensors_keep_projected = np.array([
                        project_tensor(tensor, los_vec(c)) for tensor, c in zip(tensors_for_lightcone_keep, coords_keep)
                          ])
                    out.create_dataset(value, data = tensors_keep_projected)

                for key, value in fields_soap.items():
                    data_array = soap_data[key][:]
                    data_array_for_lightcone = data_array[lightcone_to_soap_data]
                    out.create_dataset(value, data = data_array_for_lightcone[keep_lightcone])

                for key, value in fields_lightcone.items():
                    data_array = lightcone_shell[key][:]
                    out.create_dataset(value, data = data_array[keep_lightcone])

print("DONE")
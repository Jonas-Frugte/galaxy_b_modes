import numpy as np
import h5py

sim_name = "L1_m9"

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

for i in range(42, 43):
    with h5py.File(f"data/{sim_name}/shells/shell_{i:04d}.hdf5", "r") as shell:
        soap_idx = shell["SOAP_indexes"][:]
        tensors  = np.load(f"data/{sim_name}/stellar_inertia_tensors/stellar_inertia_tensors_{i:02d}.npy")

        resolved_rows = np.flatnonzero(~np.all(tensors == 0, axis=1))
        keep = np.flatnonzero(np.isin(soap_idx, resolved_rows))   # resolved object positions

        coords    = shell["halo_coords"][keep]      # h5py fancy-index reads only these rows
        masses    = shell["masses"][keep]
        redshifts = shell["redshifts"][keep]
        soap_idx  = soap_idx[keep]

    projected = np.array([project_tensor(tensors[s], los_vec(c))
                          for s, c in zip(soap_idx, coords)])

    with h5py.File(f"data/{sim_name}/shells_resolved/shell_{i:04d}.hdf5", "w") as out:
        out.create_dataset("halo_coords", data=coords)
        out.create_dataset("masses", data=masses)
        out.create_dataset("redshifts", data=redshifts)
        out.create_dataset("SOAP_indexes", data=soap_idx)
        out.create_dataset("projected_tensors", data=projected)
    
    print(f"Done with shell {i}. Num of unique galaxies: {len(resolved_rows)}.")

print("DONE")
import hdfstream
import numpy as np
import swiftsimio as sw

sim_name = "L1_m9"

root_dir = hdfstream.open("cosma", "/")
lc_dir = root_dir["FLAMINGO/" + sim_name + "/" + sim_name + "/SOAP-HBT"]

stellar_inertia_tensors = []
output_dir = "data/" + sim_name + "/stellar_inertia_tensors/" # TODO: change to makedir if doens't exist

for i in range(78):
    file_name = f"halo_properties_{i:04d}.hdf5"
    file = lc_dir[file_name]

    arr = file["BoundSubhalo/StellarInertiaTensor"][:]
    np.save(output_dir + f"stellar_inertia_tensors_{i:02d}.npy", arr)

print("Done")

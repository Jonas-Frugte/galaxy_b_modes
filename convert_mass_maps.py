import hdfstream
import numpy as np
import swiftsimio as sw
import healpy as hp
import h5py

sim_name = "L1_m9"

root_dir = hdfstream.open("cosma", "/")
lc_dir = root_dir["FLAMINGO/" + sim_name + "/" + sim_name + "/healpix_maps/nside_4096/lightcone0_shells/"]

stellar_inertia_tensors = []

nside_downsampled = 1024
output_dir = f"data/mass_maps_{nside_downsampled}/" # TODO: change to makedir if doens't exist


for i in range(60): # TODO: adjust range
    file_name = f"shell_{i}/swift_lightcone0.shell_{i}.0.hdf5"
    file = lc_dir[file_name]

    shell = file["Shell"]                      # streamed group
    output_file = h5py.File(output_dir + f"map_{i}.hdf5", "w")

    output_file.create_dataset("mass_density", data=np.load(output_dir + f"map_{i}.npy"))

    g = output_file.require_group(f"shell_info")  # create-or-get; safe on re-run
    for key in shell.attrs.keys():
        g.attrs[key] = shell.attrs[key]        # copy each attribute over

    output_file.close()

print("Done")
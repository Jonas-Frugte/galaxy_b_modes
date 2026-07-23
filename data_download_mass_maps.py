import hdfstream
import numpy as np
import swiftsimio as sw
import healpy as hp

sim_name = "L1_m9"

root_dir = hdfstream.open("cosma", "/")
lc_dir = root_dir["FLAMINGO/" + sim_name + "/" + sim_name + "/healpix_maps/nside_4096/lightcone0_shells/"]

stellar_inertia_tensors = []

nside_downsampled = 1024
output_dir = f"data/mass_maps_{nside_downsampled}/" # TODO: change to makedir if doens't exist

# TODO: make consistent with convert_mass_maps.py
for i in range(42, 60): # TODO: adjust range
    file_name = f"shell_{i}/swift_lightcone0.shell_{i}.0.hdf5"
    file = lc_dir[file_name]

    mass_map = file["TotalMass"][:]
    mass_map_downsampled = hp.ud_grade(mass_map, nside_out=nside_downsampled).astype(np.float32)
    np.save(output_dir + f"map_{i}.npy", mass_map_downsampled)

print("Done")
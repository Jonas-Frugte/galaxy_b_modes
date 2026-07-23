import hdfstream
import numpy as np
import h5py
import os

sim_name = "L1_m9"
root_dir = hdfstream.open("cosma", "/")
lc_dir = root_dir["FLAMINGO/" + sim_name + "/" + sim_name + "/halo_lightcone/lightcone0"]

fields = {
    "BoundSubhalo/TotalMass":   "masses",
    "Lightcone/HaloCentre":     "halo_coords",
    "InputHalos/SOAPIndex":     "SOAP_indexes",
    "Lightcone/SnapshotNumber": "snapshot_numbers",
    "Lightcone/Redshift":       "redshifts",
}

output_dir = "data/" + sim_name + "/shells/"
os.makedirs(output_dir, exist_ok=True)

for i in range(42, 79):
    shell_path = output_dir + f"shell_{i:04d}.hdf5"
    if os.path.exists(shell_path):          # already downloaded -> skip
        print(f"shell {i} already done, skipping")
        continue

    file = lc_dir[f"lightcone_halos_{i:04d}.hdf5"]

    tmp_path = shell_path + ".tmp"           # write to a temp name first
    with h5py.File(tmp_path, "w") as out:
        for path, name in fields.items():
            out.create_dataset(name, data=file[path][:])
    os.rename(tmp_path, shell_path)          # rename only once fully written

    print(f"shell {i} done")

print("All shells downloaded.")
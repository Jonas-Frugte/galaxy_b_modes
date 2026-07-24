"""Download FLAMINGO halo-lightcone shells.

For each lightcone shell, streams the selected halo fields off COSMA and writes
one HDF5 file per shell. Dataset unit attributes (CGS factors, a/h-scale
exponents) are copied over so the outputs stay self-describing.

Writes atomically (tmp + rename) and skips shells already downloaded, so the
script is safe to resume.
"""

import hdfstream
import h5py

import config

root_dir = hdfstream.open("cosma", "/")
lc_dir = root_dir["FLAMINGO/L2p8_m9/L2p8_m9/halo_lightcone/lightcone0"]

fields = {
    "BoundSubhalo/TotalMass":   "masses",
    "Lightcone/HaloCentre":     "halo_coords",
    "InputHalos/SOAPIndex":     "SOAP_indexes",
    "Lightcone/SnapshotNumber": "snapshot_numbers",
    "Lightcone/Redshift":       "redshifts",
}

n_shells = 79  # TODO: confirm number of lightcone_halos_*.hdf5 files

output_dir = config.LIGHTCONE_L2p8_m9
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(n_shells):
    shell_path = output_dir / f"shell_{i:04d}.hdf5"
    if shell_path.exists():                 # already downloaded -> skip
        print(f"shell {i} already done, skipping")
        continue

    file = lc_dir[f"lightcone_halos_{i:04d}.hdf5"]

    tmp_path = shell_path.with_name(shell_path.name + ".tmp")  # write temp first
    with h5py.File(tmp_path, "w") as out:
        for path, name in fields.items():
            src = file[path]
            dset = out.create_dataset(name, data=src[:])
            for key in src.attrs.keys():    # carry unit metadata over
                dset.attrs[key] = src.attrs[key]

    tmp_path.rename(shell_path)             # rename only once fully written
    print(f"shell {i} done")

print("All shells downloaded.")
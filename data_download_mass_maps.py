"""Download FLAMINGO mass-map lightcone shells (nside 4096, no downsampling).

For each shell, streams the TotalMass HEALPix map and the shell metadata off
COSMA and writes one self-contained HDF5 file per shell:

    map_{i}.hdf5
        /mass_density         the HEALPix TotalMass map for shell i
        /shell_info (group)   shell attributes (z / comoving-distance bounds, etc.)

Output directory comes from config (MASS_MAP_L2p8_m9_4096).
"""

import hdfstream
import numpy as np
import h5py

import config as cfg

sim_name = "L2p8_m9"
nside = 4096
n_shells = 68

# Remote location on COSMA (streamed, not copied wholesale).
root_dir = hdfstream.open("cosma", "/")
lc_dir = root_dir[
    f"FLAMINGO/{sim_name}/{sim_name}/healpix_maps/nside_{nside}/lightcone0_shells/"
]

# Local output directory (bs39 scratch, per config).
output_dir = cfg.MASS_MAP_L2p8_m9_4096
output_dir.mkdir(parents=True, exist_ok=True)

for i in range(n_shells):
    remote_name = f"shell_{i}/swift_lightcone0.shell_{i}.0.hdf5"
    shell_file = lc_dir[remote_name]

    mass_map = shell_file["TotalMass"][:]      # HEALPix map for this shell
    shell = shell_file["Shell"]                # group holding shell metadata

    out_path = output_dir / f"map_{i}.hdf5"
    with h5py.File(out_path, "w") as out:
        out.create_dataset("total_mass", data=mass_map)

        info = out.require_group("shell_info")
        for key in shell.attrs.keys():
            info.attrs[key] = shell.attrs[key]

    print(f"shell {i}: wrote {out_path}")

print("Done")
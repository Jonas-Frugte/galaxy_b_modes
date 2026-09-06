"""Download FLAMINGO mass-map lightcone shells (nside 4096, no downsampling).

For each shell, streams the TotalMass HEALPix map and the shell metadata off
COSMA and writes one self-contained HDF5 file per shell:

    map_{i}.hdf5
        /total_mass           the HEALPix TotalMass map for shell i
        /shell_info (group)   shell attributes (z / comoving-distance bounds, etc.)
"""

import hdfstream
import h5py

from b_modes_modules.filepaths import FilePaths

SIM_NAME = "L2p8_m9"
NSIDE = 4096


def download_mass_maps(filepaths: FilePaths, lightcone: int):
    root_dir = hdfstream.open("cosma", "/")
    lc_dir = root_dir[
        f"FLAMINGO/{SIM_NAME}/{SIM_NAME}/healpix_maps/nside_{NSIDE}/lightcone{lightcone}_shells/"
    ]
    output_dir = filepaths.MASS_MAP
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(filepaths.NSHELL_MASS_MAPS):
        out_path = output_dir / f"map_{i}.hdf5"
        if out_path.exists():
            print(f"  shell {i} already exists, skipping")
            continue

        remote_name = f"shell_{i}/swift_lightcone{lightcone}.shell_{i}.0.hdf5"
        shell_file = lc_dir[remote_name]

        mass_map = shell_file["TotalMass"][:]      # HEALPix map for this shell
        shell = shell_file["Shell"]                # group holding shell metadata

        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with h5py.File(tmp_path, "w") as out:
            out.create_dataset("total_mass", data=mass_map)
            info = out.require_group("shell_info")
            for key in shell.attrs.keys():
                info.attrs[key] = shell.attrs[key]
        tmp_path.rename(out_path)

        print(f"  shell {i}: wrote {out_path}")

    print(f"  lightcone {lightcone} mass maps done")


if __name__ == "__main__":
    for lc in range(8):
        print(f"--- lightcone {lc} ---")
        download_mass_maps(FilePaths(CAT_NAME=f"real_cat_{lc}"), lc)

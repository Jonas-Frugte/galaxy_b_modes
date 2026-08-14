"""Download FLAMINGO halo-lightcone shells.

For each lightcone shell, streams the selected halo fields off COSMA and writes
one HDF5 file per shell. Dataset unit attributes (CGS factors, a/h-scale
exponents) are copied over so the outputs stay self-describing.

Note on the two index fields, which are different quantities:
  - InputHalos/SOAPIndex           row number into the SOAP file for this
                                   shell's snapshot. This is what maps a
                                   lightcone halo onto its SOAP properties.
  - InputHalos/HaloCatalogueIndex  index into the HBT-HERONS catalogue. Pairs
                                   with SOAP's field of the same name; kept as
                                   an independent check on the row mapping.

Writes atomically (tmp + rename) and skips shells already downloaded. If a
shell's file already exists but is missing a field that was since added to
`fields` (e.g. TrackId), the file is patched in place instead of being fully
re-downloaded.
"""

import hdfstream
import h5py

import config

root_dir = hdfstream.open("cosma", "/")
lc_dir = root_dir["FLAMINGO/L2p8_m9/L2p8_m9/halo_lightcone/lightcone0"]

# HDF5 path in the source file -> dataset name in the output file.
fields = {
    # Passed through from SOAP (units: 1e10 Msun).
    "BoundSubhalo/TotalMass":          "masses",
    # Position relative to the observer. NOT InputHalos/HaloCentre, which is
    # the position in the snapshot and would give wrong lines of sight.
    "Lightcone/HaloCentre":            "halo_coords",
    "Lightcone/SnapshotNumber":        "snapshot_numbers",
    "Lightcone/Redshift":              "redshifts",
    # Row number into the SOAP catalogue for this shell's snapshot.
    "InputHalos/SOAPIndex":            "SOAP_indexes",
    # HBT-HERONS catalogue index; matches SOAP's halo_catalogue_index.
    "InputHalos/HaloCatalogueIndex":   "halo_catalogue_index",
    # Persistent track ID. Periodic replications of the same halo share a
    # TrackId, so this is what identifies replicated copies in the lightcone.
    "InputHalos/HBTplus/TrackId":      "track_id",
    # Enable if you want to split centrals from satellites later:
    "InputHalos/IsCentral":          "is_central",
}

# Global metadata groups to copy verbatim (attributes only), if present.
meta_groups = ["Units", "Cosmology"]

n_shells = 79

output_dir = config.LIGHTCONE
output_dir.mkdir(parents=True, exist_ok=True)


def download_field(src_file, path, name, out):
    """Stream one field from the source file into an open output file."""
    src = src_file[path]
    dset = out.create_dataset(name, data=src[:])
    for key in src.attrs.keys():            # carry unit metadata over
        dset.attrs[key] = src.attrs[key]


def copy_meta_groups(src_file, out):
    """Copy attribute-only metadata groups, skipping any that are absent."""
    for gname in meta_groups:
        try:
            src_group = src_file[gname]
        except KeyError:
            continue
        g = out.require_group(gname)
        for key in src_group.attrs.keys():
            g.attrs[key] = src_group.attrs[key]


for i in range(n_shells):
    shell_path = output_dir / config.SHELL_NAME(i)

    if shell_path.exists():
        # Check which requested fields are already present.
        with h5py.File(shell_path, "r") as out:
            missing = {path: name for path, name in fields.items()
                       if name not in out}
            missing_meta = [g for g in meta_groups if g not in out]

        if not missing and not missing_meta:
            print(f"shell {i} already done, skipping")
            continue

        # Patch in just what's missing, in place.
        print(f"shell {i}: patching in {list(missing.values())}")
        file = lc_dir[f"lightcone_halos_{i:04d}.hdf5"]
        with h5py.File(shell_path, "a") as out:
            for path, name in missing.items():
                download_field(file, path, name, out)
            if missing_meta:
                copy_meta_groups(file, out)
        print(f"shell {i} patched")
        continue

    # Fresh download.
    file = lc_dir[f"lightcone_halos_{i:04d}.hdf5"]

    tmp_path = shell_path.with_name(shell_path.name + ".tmp")  # write temp first
    with h5py.File(tmp_path, "w") as out:
        for path, name in fields.items():
            download_field(file, path, name, out)
        copy_meta_groups(file, out)

    tmp_path.rename(shell_path)             # rename only once fully written
    print(f"shell {i} done")

print("DONE")
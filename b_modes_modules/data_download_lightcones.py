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

from b_modes_modules.filepaths import FilePaths

FIELDS = {
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
    "InputHalos/IsCentral":            "is_central",
}
META_GROUPS = ["Units", "Cosmology"]


def download_field(src_file, path, name, out):
    """Stream one field from the source file into an open output file."""
    src = src_file[path]
    dset = out.create_dataset(name, data=src[:])
    for key in src.attrs.keys():            # carry unit metadata over
        dset.attrs[key] = src.attrs[key]


def copy_meta_groups(src_file, out):
    """Copy attribute-only metadata groups, skipping any that are absent."""
    for gname in META_GROUPS:
        try:
            src_group = src_file[gname]
        except KeyError:
            continue
        g = out.require_group(gname)
        for key in src_group.attrs.keys():
            g.attrs[key] = src_group.attrs[key]


def download_lightcone(filepaths: FilePaths, lightcone: int):
    root_dir = hdfstream.open("cosma", "/")
    lc_dir = root_dir[f"FLAMINGO/{filepaths.BOX_NAME}/{filepaths.BOX_NAME}/halo_lightcone/lightcone{lightcone}"]

    output_dir = filepaths.RAW_LIGHTCONE
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(filepaths.NSHELLS_LIGHTCONE):
        shell_path = output_dir / filepaths.SHELL_NAME(i)

        if shell_path.exists():
            with h5py.File(shell_path, "r") as out:
                missing = {path: name for path, name in FIELDS.items() if name not in out}
                missing_meta = [g for g in META_GROUPS if g not in out]

            if not missing and not missing_meta:
                print(f"  shell {i} already done, skipping")
                continue

            print(f"  shell {i}: patching in {list(missing.values())}")
            file = lc_dir[f"lightcone_halos_{i:04d}.hdf5"]
            with h5py.File(shell_path, "a") as out:
                for path, name in missing.items():
                    download_field(file, path, name, out)
                if missing_meta:
                    copy_meta_groups(file, out)
            print(f"  shell {i} patched")
            continue

        file = lc_dir[f"lightcone_halos_{i:04d}.hdf5"]

        tmp_path = shell_path.with_name(shell_path.name + ".tmp")
        with h5py.File(tmp_path, "w") as out:
            for path, name in FIELDS.items():
                download_field(file, path, name, out)
            copy_meta_groups(file, out)
        tmp_path.rename(shell_path)
        print(f"  shell {i} done")

    print(f"  lightcone {lightcone} download done")


if __name__ == "__main__":
    for lc in range(8):
        print(f"--- lightcone {lc} ---")
        download_lightcone(FilePaths(CAT_NAME=f"real_cat_{lc}"), lc)

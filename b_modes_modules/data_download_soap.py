"""Download FLAMINGO SOAP-HBT halo properties.

For each snapshot, streams a selected subset of SOAP halo properties off COSMA
and writes one HDF5 file per snapshot. Selected: total & stellar mass, particle
counts (total bound + per type), and all four stellar inertia tensors
(iterative/non-iterative x full/reduced), plus the halo catalogue index for
matching against the halo lightcone, plus the HBT-HERONS TrackId for matching
halo images across snapshots (replicated copies, mergers, etc.).

Everything needed to interpret the numbers is preserved:
  - each dataset keeps its own attrs (unit exponents, CGS conversion factors,
    a/h-scale exponents, description);
  - the global /Units group (internal-unit -> CGS base factors) is copied;
  - the /Cosmology group (h, scale factor, redshift, Omegas) is copied.

If a snapshot's output file already exists but is missing a field that was
since added to `fields` (e.g. TrackId), the file is patched in place instead
of being fully re-downloaded.

SOAP is shared across every lightcone/observer within a box, so this is a
box-level download -- call it once per box, not once per observer.
"""

import hdfstream
import h5py

from b_modes_modules.filepaths import FilePaths

# HDF5 path in the source file -> dataset name in the output file.
FIELDS = {
    # Masses (units: 1e10 Msun)
    "BoundSubhalo/TotalMass":                          "total_mass",
    "BoundSubhalo/StellarMass":                        "stellar_mass",
    # Particle counts (dimensionless)
    "InputHalos/NumberOfBoundParticles":               "n_bound_particles",
    "BoundSubhalo/NumberOfStarParticles":              "n_star_particles",
    "BoundSubhalo/NumberOfDarkMatterParticles":        "n_dm_particles",
    "BoundSubhalo/NumberOfGasParticles":               "n_gas_particles",
    "BoundSubhalo/NumberOfBlackHoleParticles":         "n_bh_particles",
    # Stellar inertia tensors. Shape 6, stored as upper triangle:
    # (1,1),(2,2),(3,3),(1,2),(1,3),(2,3). Only computed for >20 particles.
    # Non-reduced units: Mpc^2. Reduced: dimensionless.
    "BoundSubhalo/StellarInertiaTensor":               "stellar_inertia_tensor",
    "BoundSubhalo/StellarInertiaTensorNoniterative":   "stellar_inertia_tensor_noniterative",
    "BoundSubhalo/StellarInertiaTensorReduced":        "stellar_inertia_tensor_reduced",
    "BoundSubhalo/StellarInertiaTensorReducedNoniterative": "stellar_inertia_tensor_reduced_noniterative",
    # Index for matching to the halo lightcone (its InputHalos/SOAPIndex).
    "InputHalos/HaloCatalogueIndex":                   "halo_catalogue_index",
    # Persistent subhalo ID, consistent across snapshots. Needed to link
    # copies of the same halo across box replications / lightcone shells.
    "InputHalos/HBTplus/TrackId":                      "track_id",
}

# Global metadata groups to copy verbatim (attributes only).
META_GROUPS = ["Units", "Cosmology"]


def download_field(src_file, path, name, out):
    """Stream one field from the source file into an open output file."""
    src = src_file[path]
    dset = out.create_dataset(name, data=src[:])
    for key in src.attrs.keys():
        dset.attrs[key] = src.attrs[key]


def download_soap(filepaths: FilePaths):
    root_dir = hdfstream.open("cosma", "/")
    soap_dir = root_dir[f"FLAMINGO/{filepaths.BOX_NAME}/{filepaths.BOX_NAME}/SOAP-HBT"]

    output_dir = filepaths.SOAP
    output_dir.mkdir(parents=True, exist_ok=True)

    # one SOAP snapshot per lightcone shell index
    for i in range(filepaths.NSHELLS_LIGHTCONE):
        out_path = output_dir / f"halos_{i:04d}.hdf5"

        if out_path.exists():
            with h5py.File(out_path, "r") as out:
                missing = {path: name for path, name in FIELDS.items() if name not in out}

            if not missing:
                print(f"  snapshot {i} already done, skipping")
                continue

            print(f"  snapshot {i}: patching in {list(missing.values())}")
            file = soap_dir[f"halo_properties_{i:04d}.hdf5"]
            with h5py.File(out_path, "a") as out:
                for path, name in missing.items():
                    download_field(file, path, name, out)
            print(f"  snapshot {i} patched")
            continue

        file = soap_dir[f"halo_properties_{i:04d}.hdf5"]

        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with h5py.File(tmp_path, "w") as out:
            for path, name in FIELDS.items():
                download_field(file, path, name, out)

            for gname in META_GROUPS:
                src_group = file[gname]
                g = out.require_group(gname)
                for key in src_group.attrs.keys():
                    g.attrs[key] = src_group.attrs[key]

        tmp_path.rename(out_path)
        print(f"  snapshot {i} done")

    print(f"  SOAP download done: {output_dir}")


if __name__ == "__main__":
    download_soap(FilePaths())

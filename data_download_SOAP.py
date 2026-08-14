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
"""

import hdfstream
import h5py

import config

root_dir = hdfstream.open("cosma", "/")
soap_dir = root_dir["FLAMINGO/L2p8_m9/L2p8_m9/SOAP-HBT"]

# HDF5 path in the source file -> dataset name in the output file.
fields = {
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
meta_groups = ["Units", "Cosmology"]

n_snapshots = 79

output_dir = config.SOAP
output_dir.mkdir(parents=True, exist_ok=True)


def download_field(src_file, path, name, out):
    """Stream one field from the source file into an open output file."""
    src = src_file[path]
    dset = out.create_dataset(name, data=src[:])
    for key in src.attrs.keys():
        dset.attrs[key] = src.attrs[key]


for i in range(n_snapshots):
    out_path = output_dir / f"halos_{i:04d}.hdf5"

    if out_path.exists():
        # Check which requested fields are already present.
        with h5py.File(out_path, "r") as out:
            missing = {path: name for path, name in fields.items()
                       if name not in out}

        if not missing:
            print(f"snapshot {i} already done, skipping")
            continue

        # Patch in just the missing fields, in place.
        print(f"snapshot {i}: patching in {list(missing.values())}")
        file = soap_dir[f"halo_properties_{i:04d}.hdf5"]
        with h5py.File(out_path, "a") as out:
            for path, name in missing.items():
                download_field(file, path, name, out)
        print(f"snapshot {i} patched")
        continue

    # Fresh download.
    file = soap_dir[f"halo_properties_{i:04d}.hdf5"]

    tmp_path = out_path.with_name(out_path.name + ".tmp")  # write temp first
    with h5py.File(tmp_path, "w") as out:
        for path, name in fields.items():
            download_field(file, path, name, out)

        # Global unit / cosmology metadata as attribute-only groups.
        for gname in meta_groups:
            src_group = file[gname]
            g = out.require_group(gname)
            for key in src_group.attrs.keys():
                g.attrs[key] = src_group.attrs[key]

    tmp_path.rename(out_path)               # rename only once fully written
    print(f"snapshot {i} done")

print("DONE")
"""Diagnostic only: verify the lightcone -> SOAP row mapping.

The lightcone's SOAP_indexes column is a row number into the SOAP file for the
corresponding snapshot (attribute: "Index of the halo in the input SOAP
catalogue"). SOAP's halo_catalogue_index is an index into the HBT-HERONS
catalogue, which is a different quantity and plays no part in the mapping.

Correctness test: BoundSubhalo/TotalMass was downloaded from both the lightcone
and SOAP. Under the correct mapping the two must agree bitwise for every halo.

Also runs the old (wrong) searchsorted mapping for comparison, to confirm it
fails and to count how many rows it happened to get right.

Writes nothing. Reads only index and mass columns, not the tensors.
"""

import numpy as np
import h5py

import config

soap_dir = config.SOAP_L2p8_m9
lightcone_dir = config.LIGHTCONE_L2p8_m9

n_shells = 79
check_old_mapping = True     # set False once the new mapping is confirmed
printed_attrs = False
summary = []

hdr = (f"{'sh':>3} {'z_lc_min':>9} {'z_lc_max':>9} {'z_snap':>8} "
       f"{'n_lc':>10} {'n_soap':>11} {'maxRow':>11} {'n_massdiff':>11} "
       f"{'max|dm|':>11}")
if check_old_mapping:
    hdr += f" {'old_max|dm|':>12} {'old_rows_ok':>12}"
hdr += "  snaps"
print(hdr)

for i in range(n_shells):
    with h5py.File(lightcone_dir / f"shell_{i:04d}.hdf5", "r") as lc:
        lc_soap_row = lc["SOAP_indexes"][:]          # row number into SOAP file

        if lc_soap_row.size == 0:
            print(f"{i:>3} {'empty':>9}")
            continue

        z_lc = lc["redshifts"][:]
        snaps = np.unique(lc["snapshot_numbers"][:])
        lc_mass = lc["masses"][:]                     # BoundSubhalo/TotalMass
        lc_attrs = dict(lc["SOAP_indexes"].attrs)

        with h5py.File(soap_dir / f"halos_{i:04d}.hdf5", "r") as soap:
            soap_hbt_idx = soap["halo_catalogue_index"][:]   # HBT index, not a row
            z_snap = float(np.atleast_1d(soap["Cosmology"].attrs["Redshift"])[0])

            if not printed_attrs:
                print("\n--- lightcone SOAP_indexes attrs ---")
                for k, v in lc_attrs.items():
                    print(f"    {k}: {v}")
                print("--- soap datasets available ---")
                print("   ", list(soap.keys()))
                print()
                printed_attrs = True

            # --- mapping: the row number is already in the lightcone file ---
            if lc_soap_row.max() >= soap_hbt_idx.size:
                print(f"{i:>3}  ERROR: max row {lc_soap_row.max()} >= "
                      f"n_soap {soap_hbt_idx.size}")
                summary.append((i, -1, lc_soap_row.size, np.nan))
                continue

            soap_mass = soap["total_mass"][:]
            dm = np.abs(lc_mass - soap_mass[lc_soap_row])
            n_massdiff = int((dm > 0).sum())

            line = (f"{i:>3} {z_lc.min():>9.4f} {z_lc.max():>9.4f} {z_snap:>8.4f} "
                    f"{lc_soap_row.size:>10d} {soap_hbt_idx.size:>11d} "
                    f"{lc_soap_row.max():>11d} {n_massdiff:>11d} {dm.max():>11.4g}")

            # --- old mapping, for comparison ---
            if check_old_mapping:
                sorter = np.argsort(soap_hbt_idx)
                old = sorter[np.searchsorted(soap_hbt_idx, lc_soap_row,
                                             sorter=sorter)]
                old = np.minimum(old, soap_hbt_idx.size - 1)
                dm_old = np.abs(lc_mass - soap_mass[old])
                rows_ok = int((old == lc_soap_row).sum())
                line += f" {dm_old.max():>12.4g} {rows_ok:>12d}"

            line += f"  {snaps}"
            print(line)

            # --- detail on the first shell with any mass disagreement ---
            if n_massdiff and not any(s[1] > 0 for s in summary):
                bad = dm > 0
                print(f"\n=== shell {i}: mass disagreement under new mapping ===")
                print(f"  n disagreeing : {n_massdiff} of {dm.size} "
                      f"({n_massdiff / dm.size:.4%})")
                print(f"  max |dm|      : {dm.max():.6g}")
                print(f"  rows involved min/max : "
                      f"{lc_soap_row[bad].min()} / {lc_soap_row[bad].max()}")
                print(f"  lc_mass   sample : {lc_mass[bad][:5]}")
                print(f"  soap_mass sample : {soap_mass[lc_soap_row][bad][:5]}")
                print(f"  duplicate rows in lightcone : "
                      f"{lc_soap_row.size - np.unique(lc_soap_row).size}")
                print()

            summary.append((i, n_massdiff, lc_soap_row.size, float(dm.max())))

print("\n=== summary ===")
ok = [s for s in summary if s[1] == 0]
bad = [s for s in summary if s[1] > 0]
err = [s for s in summary if s[1] < 0]
tot = sum(s[2] for s in summary if s[1] >= 0)
tot_bad = sum(s[1] for s in bad)
print(f"shells processed          : {len(summary)}")
print(f"shells fully consistent   : {len(ok)}")
print(f"shells with mass mismatch : {len(bad)}")
print(f"shells with bad row range : {len(err)}")
if tot:
    print(f"halos with mass mismatch  : {tot_bad} of {tot} ({tot_bad / tot:.4%})")
if bad:
    print(f"affected shells           : {[s[0] for s in bad]}")
    print(f"worst max|dm|             : {max(s[3] for s in bad):.6g}")
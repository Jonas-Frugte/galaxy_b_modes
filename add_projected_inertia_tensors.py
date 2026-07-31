"""Diagnostic only: why do some lightcone halos fail to match a SOAP row?

Writes nothing. Runs over all shells, reports per-shell matching statistics and
the properties of the unmatched halos, then prints a summary.

Run on the head node; it only reads index/mass columns, not the tensors.
"""

import numpy as np
import h5py

import config

soap_dir = config.SOAP_L2p8_m9
lightcone_dir = config.LIGHTCONE_L2p8_m9

n_shells = 79
printed_attrs = False
summary = []

print(f"{'sh':>3} {'z_lc_min':>9} {'z_lc_max':>9} {'z_snap':>8} "
      f"{'n_lc':>10} {'n_soap':>11} {'unmatched':>10} {'frac':>8} "
      f"{'maxSOAPIdx':>12} {'maxCatIdx':>12} {'snaps'}")

for i in range(n_shells):
    with h5py.File(lightcone_dir / f"shell_{i:04d}.hdf5", "r") as lc:
        soap_idx_lc = lc["SOAP_indexes"][:]

        if soap_idx_lc.size == 0:
            print(f"{i:>3} {'empty':>9}")
            continue

        z_lc = lc["redshifts"][:]
        snaps = np.unique(lc["snapshot_numbers"][:])
        lc_mass = lc["masses"][:]
        lc_attrs = dict(lc["SOAP_indexes"].attrs)

        with h5py.File(soap_dir / f"halos_{i:04d}.hdf5", "r") as soap:
            soap_idx = soap["halo_catalogue_index"][:]
            z_snap = float(np.atleast_1d(soap["Cosmology"].attrs["Redshift"])[0])
            soap_attrs = dict(soap["halo_catalogue_index"].attrs)

            # --- attributes, once ---
            if not printed_attrs:
                print("\n--- lightcone SOAP_indexes attrs ---")
                for k, v in lc_attrs.items():
                    print(f"    {k}: {v}")
                print("--- soap halo_catalogue_index attrs ---")
                for k, v in soap_attrs.items():
                    print(f"    {k}: {v}")
                print("--- soap datasets available ---")
                print("   ", list(soap.keys()))
                print()
                printed_attrs = True

            # --- the lookup, clamped so nothing raises ---
            is_sorted = bool(np.all(soap_idx[:-1] < soap_idx[1:]))
            if is_sorted:
                pos = np.searchsorted(soap_idx, soap_idx_lc)
            else:
                sorter = np.argsort(soap_idx)
                pos = sorter[np.searchsorted(soap_idx, soap_idx_lc, sorter=sorter)]
            pos = np.minimum(pos, soap_idx.size - 1)

            matched = soap_idx[pos] == soap_idx_lc
            n_bad = int((~matched).sum())
            frac_bad = n_bad / matched.size

            print(f"{i:>3} {z_lc.min():>9.4f} {z_lc.max():>9.4f} {z_snap:>8.4f} "
                  f"{soap_idx_lc.size:>10d} {soap_idx.size:>11d} "
                  f"{n_bad:>10d} {frac_bad:>8.4%} "
                  f"{soap_idx_lc.max():>12d} {soap_idx.max():>12d} {snaps}")

            # --- deeper look on the first shell that has unmatched halos ---
            if n_bad and len(summary) == 0 or (n_bad and not any(s[1] for s in summary)):
                bad_ids = soap_idx_lc[~matched]
                bad_mass = lc_mass[~matched]
                good_mass = lc_mass[matched]

                print(f"\n=== shell {i}: first shell with unmatched halos ===")
                print(f"  soap_idx sorted strictly increasing : {is_sorted}")
                print(f"  soap_idx has duplicates             : "
                      f"{soap_idx.size != np.unique(soap_idx).size}")
                print(f"  lightcone IDs present in soap ID set: "
                      f"{int(np.isin(soap_idx_lc, soap_idx).sum())} of {soap_idx_lc.size}")
                print(f"  unmatched IDs   min/max : {bad_ids.min()} / {bad_ids.max()}")
                print(f"  soap IDs        min/max : {soap_idx.min()} / {soap_idx.max()}")
                print(f"  unmatched IDs below soap min : {int((bad_ids < soap_idx.min()).sum())}")
                print(f"  unmatched IDs above soap max : {int((bad_ids > soap_idx.max()).sum())}")
                print(f"  unmatched mass [1e10 Msun] min/median/max : "
                      f"{bad_mass.min():.4g} / {np.median(bad_mass):.4g} / {bad_mass.max():.4g}")
                if good_mass.size:
                    print(f"  matched   mass [1e10 Msun] min/median/max : "
                          f"{good_mass.min():.4g} / {np.median(good_mass):.4g} / {good_mass.max():.4g}")
                # is SOAPIndex plausibly a row number rather than a catalogue ID?
                print(f"  max SOAPIndex vs n_soap_rows : "
                      f"{soap_idx_lc.max()} vs {soap_idx.size}")
                print(f"  first 10 unmatched IDs : {bad_ids[:10]}")
                print(f"  first 10 soap IDs      : {soap_idx[:10]}")
                print()

            summary.append((i, n_bad, matched.size))

print("\n=== summary ===")
tot_bad = sum(s[1] for s in summary)
tot_all = sum(s[2] for s in summary)
n_shells_bad = sum(1 for s in summary if s[1])
print(f"shells processed        : {len(summary)}")
print(f"shells with unmatched   : {n_shells_bad}")
print(f"total unmatched halos   : {tot_bad} of {tot_all} ({tot_bad / tot_all:.4%})")
if n_shells_bad:
    print(f"shells affected         : {[s[0] for s in summary if s[1]]}")
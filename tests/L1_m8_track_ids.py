import hdfstream
import numpy as np

root_dir = hdfstream.open("cosma", "/")
lc_file = root_dir["FLAMINGO/L1_m8/L1_m8/halo_lightcone/lightcone0/lightcone_halos_0006.hdf5"]
soap_file = root_dir["FLAMINGO/L1_m8/L1_m8/SOAP-HBT/halo_properties_0006.hdf5"]

N = 100_000
soap_id = lc_file["InputHalos"]["SOAPIndex"][:N]
hbt_trackid_lc = lc_file["InputHalos/HBTplus"]["TrackId"][:N]
hbt_trackid_soap = soap_file["InputHalos/HBTplus"]["TrackId"][soap_id]

print("using the PROVIDED SOAPIndex:")
print(" ", np.all(hbt_trackid_lc == hbt_trackid_soap))

# --- now construct our own soap_idx from TrackIds directly ---

# Step 1: pull the *entire* SOAP TrackId array for this snapshot (not just the
# rows soap_id happens to point at) -- we need the full set of (row -> TrackId)
# pairs so we can search it for arbitrary TrackIds.
full_soap_trackid = soap_file["InputHalos/HBTplus"]["TrackId"][:]
print(f"\nfull SOAP snapshot has {len(full_soap_trackid):,} rows")

# Step 2: sort it, but keep track of which original row each sorted value came
# from. `order` is the permutation that sorts full_soap_trackid; sorted_tid is
# the actual sorted values. So sorted_tid[k] == full_soap_trackid[order[k]] --
# order[k] tells you the ORIGINAL row of the value now sitting at sorted
# position k.
order = np.argsort(full_soap_trackid)
sorted_tid = full_soap_trackid[order]

# Step 3: for each lightcone TrackId, binary-search sorted_tid for where it
# WOULD be inserted to keep things sorted. If that TrackId actually exists,
# this position is exactly where it sits.
pos = np.searchsorted(sorted_tid, hbt_trackid_lc)

# Step 4: searchsorted doesn't check that the value is *actually* there -- it
# just gives you an insertion point, even for values that don't exist (they'd
# insert between two unrelated neighbours). Also, a value larger than
# everything in sorted_tid would want to insert at position len(sorted_tid),
# which is out of bounds. So: clip to a valid index, then explicitly check the
# value at that position really matches.
pos_clipped = np.clip(pos, 0, len(sorted_tid) - 1)
found = (pos < len(sorted_tid)) & (sorted_tid[pos_clipped] == hbt_trackid_lc)

# Step 5: convert "position in the sorted array" back to "row in the original
# (unsorted) SOAP file" via order. This IS our reconstructed soap_idx.
reconstructed_soap_id = np.full(N, -1, dtype=np.int64)
reconstructed_soap_id[found] = order[pos_clipped[found]]

print(f"lightcone TrackIds with no match in SOAP at all: {(~found).sum():,}")

print("\nusing OUR RECONSTRUCTED soap_idx:")
hbt_trackid_soap_reconstructed = full_soap_trackid[reconstructed_soap_id[found]]
print(" ", np.all(hbt_trackid_soap_reconstructed == hbt_trackid_lc[found]))

agree_with_provided = reconstructed_soap_id[found] == soap_id[found]
print(f"\nreconstructed idx matches provided SOAPIndex for "
      f"{agree_with_provided.sum():,}/{found.sum():,} rows "
      f"({agree_with_provided.mean()*100:.2f}%)")

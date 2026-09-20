"""One-off: convert already-downloaded pot_der_alms shell files from the old
5-dataset format (grad/kappa/gammaE/F/G alms) to the new grav_pot_alms-only
format, by inverting kappa_alms. No mass maps or network access needed --
this only touches files already sitting on disk.

Self-validating: before overwriting a shell, re-derives the other four
quantities from the reconstructed grav_pot_alms and checks them against
what's actually stored. Aborts (raises) on the first shell that doesn't
check out, rather than silently overwriting good data with something wrong.

Usage: python3 convert_alms_to_potential_only.py [box_name] [first_lc] [last_lc]
"""
import sys
import h5py
import healpy as hp
import numpy as np

from b_modes_modules.filepaths import FilePaths
from b_modes_modules.gen_pot_alms import grav_pot_alms_from_kappa, derived_alms_from_potential

BOX_NAME = sys.argv[1] if len(sys.argv) > 1 else "L2p8_m9"
FIRST_LC = int(sys.argv[2]) if len(sys.argv) > 2 else 0
LAST_LC = int(sys.argv[3]) if len(sys.argv) > 3 else 7

RTOL = 1e-3  # generous margin above expected float32 rounding noise


def relative_error(a, b):
    denom = np.maximum(np.abs(b), 1e-30)
    return np.max(np.abs(a - b) / denom)


for lc in range(FIRST_LC, LAST_LC + 1):
    filepaths = FilePaths(BOX_NAME=BOX_NAME, CAT_NAME=f"real_cat_{lc}")
    alm_dir = filepaths.POT_DER_ALMS
    if not alm_dir.exists():
        print(f"observer {lc}: {alm_dir} doesn't exist, skipping")
        continue

    if not filepaths.CHIS_MASS_MAP.exists():
        print(f"observer {lc}: WARNING -- {filepaths.CHIS_MASS_MAP} is missing. "
              f"This script doesn't touch it (would need the mass maps); "
              f"check separately whether it needs regenerating.")

    n_converted = n_skipped = 0
    for shell_path in sorted(alm_dir.glob("shell_*.hdf5")):
        with h5py.File(shell_path, "r") as f:
            if "kappa_alms" not in f:
                if "grav_pot_alms" in f:
                    n_skipped += 1
                else:
                    print(f"  {shell_path.name}: unrecognized format (neither "
                          f"kappa_alms nor grav_pot_alms present), skipping")
                continue

            kappa_alms = f["kappa_alms"][:]
            grad_alms = f["grad_alms"][:]
            gammaE_alms = f["gammaE_alms"][:]
            F_alms = f["F_alms"][:]
            G_alms = f["G_alms"][:]
            shell_index = f.attrs.get("shell_index")

        lmax = hp.Alm.getlmax(len(kappa_alms))
        grav_pot_alms = grav_pot_alms_from_kappa(kappa_alms)

        # self-check: re-derive everything from the reconstructed potential
        # and confirm it matches what was actually stored, before trusting it
        grad2, kappa2, gammaE2, F2, G2 = derived_alms_from_potential(grav_pot_alms, lmax)
        errs = {
            "grad": relative_error(grad2, grad_alms),
            "kappa": relative_error(kappa2, kappa_alms),
            "gammaE": relative_error(gammaE2, gammaE_alms),
            "F": relative_error(F2, F_alms),
            "G": relative_error(G2, G_alms),
        }
        worst = max(errs.values())
        if worst > RTOL:
            raise RuntimeError(
                f"{shell_path}: reconstruction check failed, max relative error "
                f"{worst:.2e} (per-field: {errs}) -- aborting without overwriting"
            )

        tmp_path = shell_path.with_name(shell_path.name + ".tmp")
        with h5py.File(tmp_path, "w") as out:
            out.create_dataset("grav_pot_alms", data=grav_pot_alms.astype(np.complex64))
            if shell_index is not None:
                out.attrs["shell_index"] = shell_index
        tmp_path.rename(shell_path)
        n_converted += 1
        print(f"  {shell_path.name}: converted (max reconstruction error {worst:.2e})")

    print(f"observer {lc}: {n_converted} converted, {n_skipped} already done")

print("all done")

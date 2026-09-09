"""For each observer in a box: download its raw lightcone shells and mass
maps, process them into pot_der_alms (small) and resolved subhalos, then
delete the raw lightcone + mass maps before moving to the next observer.
Keeps peak disk usage to roughly one observer's raw data at a time.

SOAP is shared across every observer in a box, so it's downloaded once up
front (if not already present) rather than per observer.

Usage: python3 process_observers.py [box_name] [first_lc] [last_lc] \
           [n_shell_mass_maps] [n_shells_lightcone]
  last_lc is inclusive. Defaults match the original L2p8_m9 run
  (observer 0 already done separately, so it's skipped by default).
"""
import shutil
import sys

from b_modes_modules.filepaths import FilePaths
from b_modes_modules.lens_spec import LensSpec
from b_modes_modules import gen_pot_alms
from b_modes_modules.data_download_lightcones import download_lightcone
from b_modes_modules.data_download_mass_maps import download_mass_maps
from b_modes_modules.data_download_soap import download_soap
from b_modes_modules.add_projected_inertia_tensors import process_resolved_subhalos

BOX_NAME = sys.argv[1] if len(sys.argv) > 1 else "L2p8_m9"
FIRST_LC = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LAST_LC = int(sys.argv[3]) if len(sys.argv) > 3 else 7
N_SHELL_MASS_MAPS = int(sys.argv[4]) if len(sys.argv) > 4 else 68
N_SHELLS_LIGHTCONE = int(sys.argv[5]) if len(sys.argv) > 5 else 79

soap_check = FilePaths(BOX_NAME=BOX_NAME, NSHELLS_LIGHTCONE=N_SHELLS_LIGHTCONE)
if not any(soap_check.SOAP.glob("*.hdf5")):
    print(f"=== no SOAP data found for {BOX_NAME}, downloading ===")
    download_soap(soap_check)

for lc in range(FIRST_LC, LAST_LC + 1):
    filepaths = FilePaths(BOX_NAME=BOX_NAME, CAT_NAME=f"real_cat_{lc}",
                           NSHELL_MASS_MAPS=N_SHELL_MASS_MAPS, NSHELLS_LIGHTCONE=N_SHELLS_LIGHTCONE)
    lens_spec = LensSpec()

    print(f"=== observer {lc}: downloading lightcone ===")
    download_lightcone(filepaths, lc)

    print(f"=== observer {lc}: downloading mass maps ===")
    download_mass_maps(filepaths, lc)

    print(f"=== observer {lc}: computing pot_der_alms ===")
    gen_pot_alms.process_catalogue(filepaths, lens_spec)

    print(f"=== observer {lc}: building resolved subhalos ===")
    process_resolved_subhalos(filepaths, lens_spec)

    print(f"=== observer {lc}: deleting raw lightcone + mass maps ===")
    shutil.rmtree(filepaths.RAW_LIGHTCONE, ignore_errors=True)
    shutil.rmtree(filepaths.MASS_MAP, ignore_errors=True)

    print(f"=== observer {lc}: done ===")

print("all observers processed")

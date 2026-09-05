"""For each observer (lightcone 0-7): download its raw lightcone shells and
mass maps, process them into pot_der_alms (small) and resolved subhalos,
then delete the raw lightcone + mass maps before moving to the next observer.
Keeps peak disk usage to roughly one observer's raw data at a time.

SOAP is shared across observers and must already be downloaded separately
(data_download_SOAP.py) before running this.
"""
import shutil

from b_modes_modules.filepaths import FilePaths
from b_modes_modules.lens_spec import LensSpec
from b_modes_modules import gen_pot_alms
from b_modes_modules.data_download_lightcones import download_lightcone
from b_modes_modules.data_download_mass_maps import download_mass_maps
from b_modes_modules.add_projected_inertia_tensors import process_resolved_subhalos

soap_check = FilePaths()
if not any(soap_check.SOAP.glob("*.hdf5")):
    raise SystemExit(f"no SOAP data found at {soap_check.SOAP} -- run data_download_SOAP.py first")

for lc in range(8):
    filepaths = FilePaths(CAT_NAME=f"real_cat_{lc}")
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

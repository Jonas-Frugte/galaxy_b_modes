from b_modes_modules.filepaths import FilePaths
from b_modes_modules.lens_spec import LensSpec
from b_modes_modules.data_download_lightcones import download_lightcone

for lc in range(8):
    print(f"--- lightcone {lc} ---")
    download_lightcone(FilePaths(CAT_NAME=f"real_cat_{lc}"), LensSpec(lightcone=lc))

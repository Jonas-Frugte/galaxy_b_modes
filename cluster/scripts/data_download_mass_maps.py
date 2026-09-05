from b_modes_modules.filepaths import FilePaths
from b_modes_modules.lens_spec import LensSpec
from b_modes_modules.data_download_mass_maps import download_mass_maps

for lc in range(8):
    print(f"--- lightcone {lc} ---")
    download_mass_maps(FilePaths(CAT_NAME=f"real_cat_{lc}"), LensSpec(lightcone=lc))

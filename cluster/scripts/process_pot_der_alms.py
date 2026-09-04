from b_modes_modules import gen_pot_alms
from b_modes_modules.filepaths import FilePaths

LIGHTCONES = range(8)

for lc in LIGHTCONES:
    filepaths = FilePaths(CAT_NAME=f"real_cat_{lc}")
    print(f"--- real_cat_{lc} ---")
    gen_pot_alms.process_catalogue(filepaths)

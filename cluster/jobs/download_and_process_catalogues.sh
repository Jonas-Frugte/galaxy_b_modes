#!/bin/bash
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate b-modes

cd "$(dirname "$0")/../scripts"

python3 -u data_download_lightcones.py
python3 -u data_download_mass_maps.py
python3 -u process_pot_der_alms.py

echo "all downloads + alm processing done"

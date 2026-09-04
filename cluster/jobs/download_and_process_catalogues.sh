#!/bin/bash
#$ -N download_and_process
#$ -o /nethome/frugt001/galaxy_b_modes/cluster/jobs/logs/download_and_process.out
#$ -e /nethome/frugt001/galaxy_b_modes/cluster/jobs/logs/download_and_process.err
#$ -q itf-fat.q
#$ -l h_rt=72:00:00
#$ -l h_vmem=64G
#$ -pe smp 1
#$ -cwd
#$ -V
#$ -m abe
#$ -M j.s.a.frugte@uu.nl

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate b-modes

cd /nethome/frugt001/galaxy_b_modes/cluster/scripts

python3 -u data_download_lightcones.py
python3 -u data_download_mass_maps.py
python3 -u process_pot_der_alms.py

echo "all downloads + alm processing done"

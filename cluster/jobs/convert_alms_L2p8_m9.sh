#!/bin/bash
#$ -N convert_alms_L2p8_m9
#$ -o /nethome/frugt001/galaxy_b_modes/cluster/jobs/logs/convert_alms_L2p8_m9.out
#$ -e /nethome/frugt001/galaxy_b_modes/cluster/jobs/logs/convert_alms_L2p8_m9.err
#$ -q itf-fat.q
#$ -l h_rt=6:00:00
#$ -l h_vmem=32G
#$ -pe smp 1
#$ -cwd
#$ -V
#$ -m abe
#$ -M j.s.a.frugte@uu.nl

set -e

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate b-modes

cd /nethome/frugt001/galaxy_b_modes/cluster/scripts

python3 -u convert_alms_to_potential_only.py L2p8_m9 0 7

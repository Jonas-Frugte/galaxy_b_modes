#!/bin/bash
#$ -N download_and_process_L1_m8
#$ -o /nethome/frugt001/galaxy_b_modes/cluster/jobs/logs/download_and_process_L1_m8.out
#$ -e /nethome/frugt001/galaxy_b_modes/cluster/jobs/logs/download_and_process_L1_m8.err
#$ -q itf-fat.q
#$ -l h_rt=72:00:00
#$ -l h_vmem=128G
#$ -pe smp 32
#$ -cwd
#$ -V
#$ -m abe
#$ -M j.s.a.frugte@uu.nl

set -e

export OMP_NUM_THREADS=$NSLOTS

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate b-modes

cd /nethome/frugt001/galaxy_b_modes/cluster/scripts

# n_shells_lightcone (79) is the L2p8_m9 default, unconfirmed for L1_m8 -- update if it differs
python3 -u process_observers.py L1_m8 0 1 60 79

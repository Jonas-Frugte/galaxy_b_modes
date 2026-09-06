#!/bin/bash
#$ -N download_and_process
#$ -o /nethome/frugt001/galaxy_b_modes/cluster/jobs/logs/download_and_process.out
#$ -e /nethome/frugt001/galaxy_b_modes/cluster/jobs/logs/download_and_process.err
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

python3 -u process_observers.py

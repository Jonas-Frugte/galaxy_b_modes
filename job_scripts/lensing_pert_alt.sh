#!/bin/bash
#$ -N lens_pert_alt
#$ -o /nethome/frugt001/galaxy_b_modes/job_scripts/lens_pert_alt.out
#$ -e /nethome/frugt001/galaxy_b_modes/job_scripts/lens_pert_alt.err
#$ -q itf-fat.q
#$ -l h_rt=12:00:00
#$ -l h_vmem=64G
#$ -pe smp 1
#$ -cwd
#$ -V
#$ -m abe
#$ -M j.s.a.frugte@uu.nl

export OMP_NUM_THREADS=$NSLOTS

source ~/miniconda3/etc/profile.d/conda.sh
conda activate b-modes
python3 -u lensing_pert_alt.py
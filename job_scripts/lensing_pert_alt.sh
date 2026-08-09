#!/bin/bash
#PBS -N lens_pert_alt
#PBS -o /nethome/frugt001/galaxy_b_modes/job_scripts/lens_pert_alt.out
#PBS -e /nethome/frugt001/galaxy_b_modes/job_scripts/lens_pert_alt.err
#PBS -l walltime=12:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=64gb
#PBS -m aeb
#PBS -M j.s.a.frugte@uu.nl

cd /nethome/frugt001/galaxy_b_modes
export OMP_NUM_THREADS=$PBS_NUM_PPN

conda activate b-modes
python3 -u lensing_pert_alt.py
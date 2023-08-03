#!/bin/bash

#SBATCH -J assessing_packing
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=24g
#SBATCH -t 01-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
module add cuda
conda activate pippack
python /proj/kuhl_lab/users/nzrandol/PIPPack/assess_packing.py \
       /proj/kuhl_lab/users/nzrandol/PIPPack/sampled_pdbs/top2018_test_pdbs \
       /proj/kuhl_lab/users/nzrandol/PIPPack/sampled_pdbs/pippack_ipmp_r3X_bbDsc_noCrop/01 \
       --out_filename=packing_stats_bf40 \
       --sc_b_factor_cutoff=40 \
       --verbose \
       --convert_mse

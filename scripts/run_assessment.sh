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
python /nas/longleaf/home/nzrandol/kuhl_lab/users/nzrandol/PIPPack/assess_packing.py \
       /nas/longleaf/home/nzrandol/kuhl_lab/users/nzrandol/PIPPack/test_pdbs \
       /nas/longleaf/home/nzrandol/kuhl_lab/users/nzrandol/PIPPack/test_pdbs_packed \
       --out_filename=packing_stats \
       --sc_b_factor_cutoff=10000 \
       --verbose \
       --convert_mse

#!/bin/bash

#SBATCH -J compute_clashscore
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32g
#SBATCH -t 00-02:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
conda activate pippack
python /proj/kuhl_lab/users/nzrandol/PIPPack/eval/compute_clashscore.py \
    --pdb_dir /proj/kuhl_lab/users/nzrandol/PIPPack/sampled_pdbs/top2018_test_pdbs/ \
    --num_workers 8

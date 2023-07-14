#!/bin/bash

#SBATCH -J compute_clashscore
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32g
#SBATCH -t 00-02:00:00
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
conda activate packergnn
python /proj/kuhl_lab/users/nzrandol/PIPPack/eval/compute_clashscore.py \
    --pdb_dir /proj/kuhl_lab/users/nzrandol/PIPPack/test_pdbs \
    --num_workers 8

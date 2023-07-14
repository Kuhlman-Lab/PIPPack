#!/bin/bash

#SBATCH -J rosetta_sc_min
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32g
#SBATCH -t 01-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
conda activate pyrosetta_env
python /proj/kuhl_lab/users/nzrandol/PIPPack/eval/rosetta_sc_min.py \
       /proj/kuhl_lab/users/nzrandol/PIPPack/test_pdbs_packed

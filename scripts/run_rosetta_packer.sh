#!/bin/bash

#SBATCH -J rosetta_packer
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=32g
#SBATCH -t 03-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
module load gcc
module load cuda
conda activate pyrosetta_env
python /proj/kuhl_lab/users/nzrandol/PIPPack/eval/rosetta_benchmark.py \
    /proj/kuhl_lab/users/nzrandol/PIPPack/test_pdbs

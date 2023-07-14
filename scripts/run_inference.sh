#!/bin/bash

#SBATCH -J inferencing_pippack
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16g
#SBATCH -t 00-01:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
module add cuda
conda activate pippack
python /nas/longleaf/home/nzrandol/kuhl_lab/users/nzrandol/PIPPack/inference.py \
       inference.exp_dir=/proj/kuhl_lab/users/nzrandol/PIPPack/exp_logs/pippack_ipmp_r3X/2023-07-14_13-05-08 \
       inference.pdb_path=/proj/kuhl_lab/users/nzrandol/PIPPack/test_pdbs \
       inference.output_dir=/proj/kuhl_lab/users/nzrandol/PIPPack/test_pdbs_packed \
       inference.seed=1234 \
       inference.n_recycle=3

# Testing setup: experiment.name=test experiment.seed=1234

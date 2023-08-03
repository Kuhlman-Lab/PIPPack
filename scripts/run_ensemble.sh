#!/bin/bash

#SBATCH -J inferencing_pippack_ensemble
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH -t 00-01:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
module add cuda
conda activate pippack
python /nas/longleaf/home/nzrandol/kuhl_lab/users/nzrandol/PIPPack/ensembled_inference.py \
       inference.pdb_path=/proj/kuhl_lab/users/nzrandol/PIPPack/sampled_pdbs/top2018_test_pdbs \
       inference.seed=1234 \
       inference.n_recycle=3

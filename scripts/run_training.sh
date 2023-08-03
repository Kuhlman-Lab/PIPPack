#!/bin/bash

#SBATCH -J training_pippack
#SBATCH -p beta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64g
#SBATCH -t 04-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
module add cuda
conda activate pippack
python /nas/longleaf/home/nzrandol/kuhl_lab/users/nzrandol/PIPPack/train.py \
       experiment=pippack_ipmp \
       experiment.name=pippack_ipmp

#!/bin/bash

#SBATCH -J training_proteinmpnn
#SBATCH -p beta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH -t 3-00:00:00
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
module add cuda
conda activate mlfold
python ./training.py \
		--path_for_outputs "./ipmp_000_03" \
		--path_for_training_data "/proj/kuhl_lab/datasets/pdb_2021aug02" \
		--num_epochs 150 \
		--save_model_every_n_epochs 1 \
		--num_neighbors 48 \
		--backbone_noise 0.0 \
		--use_ipmp

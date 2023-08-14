#!/bin/bash

#SBATCH -J testing_proteinmpnn
#SBATCH -p kuhlab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64g
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --output=slurm-%j-%x.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nzrandol@unc.edu

source ~/.bashrc
module add cuda
conda activate mlfold
python ../testing.py \
       --path_for_training_data "/proj/kuhl_lab/datasets/pdb_2021aug02" \
       --seed 1234 \
       --num_repeats 1 \
       --temperature 0.00001 \
       --num_examples_per_epoch 100000 \
       --ckpt_path ./best_ckpt_epoch133.pt
#       --use_ipmp

# --ckpt_path ./retrain_010/model_weights/epoch_last.pt 

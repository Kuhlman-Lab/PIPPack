#!/bin/bash

#SBATCH -J training_pippack
#SBATCH -p kuhlab
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=64g
#SBATCH -t 03-00:00:00
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
       experiment.name=pippack_ipmp_r3X \
       experiment.use_b_factor_mask=True \
       experiment.n_recycle=3 \
       experiment.epochs=3 \
       model.bb_dihedral=None \
       model.n_chi_bins=72 \
       dataset=top2018 \
       dataset.n_chi_bins=72 \
       dataset.num_workers=0 \
       dataset.num_chains=5 \
       dataset.batch_size=1

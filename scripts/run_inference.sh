#!/bin/bash

#SBATCH -J inferencing_pippack
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
python /nas/longleaf/home/nzrandol/kuhl_lab/users/nzrandol/PIPPack/inference.py \
       inference.model_name=pippack_model_1 \
       inference.pdb_path=/proj/kuhl_lab/users/nzrandol/PIPPack/sampled_pdbs/top2018_test_pdbs \
       inference.seed=1234 \
       inference.n_recycle=3 \
       inference.temperature=0.0 \
       inference.use_resample=True \
       inference.resample_args.sample_temp=0.1 \
       inference.resample_args.max_iters=50 \
       inference.resample_args.clash_overlap_tolerance=0.4
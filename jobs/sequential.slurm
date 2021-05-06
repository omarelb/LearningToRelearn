#!/bin/bash

#SBATCH --job-name=sequential
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00
#SBATCH --mem=20G
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm_logs/R-%x.%j.out

#Loading modules
module purge
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

# activate virtual environment
. venv/bin/activate

python train_text_cls.py learner=sequential name=sequential-test learner.samples_per_task=8000 \
                        testing.average_validation_size=32


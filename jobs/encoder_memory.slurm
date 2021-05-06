#!/bin/bash

#SBATCH --job-name=enc-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=05:00:00
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

python train_text_cls.py learner=encoder_memory name=encodermemory-test \
                         training.validate_freq=5 debug_logging=true learner.key_dim=[768,384,192,96,48]\
                         learner.samples_per_task=10000 training.log_freq=5
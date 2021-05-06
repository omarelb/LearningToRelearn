#!/bin/bash

#SBATCH --job-name=alternating
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00
#SBATCH --mem=20G
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/R-%x.%j.out

#Loading modules
module purge
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

# activate virtual environment
. venv/bin/activate

python train_text_cls.py learner=alternating name=alternating \
                        learner.data.alternating_tasks=["agnews","amazon","yahoo","dbpedia","yelp"] \
                        learner.data.alternating_n_samples_per_switch=10000 \
                        training.log_freq=1 \
                        training.validate_freq=1 \
                        name=val_during_training
                        # learner.data.alternating_relative_frequencies # How much each task is seen compared to others.

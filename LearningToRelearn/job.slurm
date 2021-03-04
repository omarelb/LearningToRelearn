#!/bin/bash

#SBATCH --job-name=hi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=5:00:00
#SBATCH --mem=20G
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1

#Loading modules
module purge
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

# activate virtual environment
. ../venv/bin/activate

python train_text_cls.py --order 1


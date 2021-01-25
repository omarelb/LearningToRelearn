#!/bin/bash
#Set job requirements
#SBATCH -t 00:01:00
#SBATCH -p gpu_short

#Loading modules
module purge
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

# activate virtual environment
. venv/bin/activate

python train_text_cls.py learner=agem

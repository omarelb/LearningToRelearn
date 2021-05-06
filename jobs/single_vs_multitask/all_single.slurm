#!/bin/bash

#SBATCH --job-name=all-single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=20:00:00
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

python train_text_cls.py learner=single learner.dataset=amazon name=single-vs-multitask-single-amazon \
                                        testing.few_shot=false learner.samples_per_task=20000
python train_text_cls.py learner=single learner.dataset=agnews name=single-vs-multitask-single-agnews \
                                        testing.few_shot=false learner.samples_per_task=20000
python train_text_cls.py learner=single learner.dataset=dbpedia name=single-vs-multitask-single-dbpedia \
                                        testing.few_shot=false learner.samples_per_task=20000
python train_text_cls.py learner=single learner.dataset=yahoo name=single-vs-multitask-single-yahoo \
                                        testing.few_shot=false learner.samples_per_task=20000


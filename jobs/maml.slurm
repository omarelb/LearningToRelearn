#!/bin/bash

#SBATCH --job-name=maml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00
#SBATCH --mem=30G
#SBATCH --partition=gpu_titanrtx_shared
#SBATCH --gres=gpu:1
#SBATCH --output=debug_log/R-%x.%j.out

#Loading modules
module purge
module load 2020
module load Python/3.8.2-GCCcore-9.3.0

# activate virtual environment
. venv/bin/activate

# python train_text_cls.py learner=anml name=debug wandb=false
python train_text_cls.py learner=maml testing.n_samples=1024 name=general-maml \
                        task_order=[agnews,dbpedia,evaluation,amazon,evaluation,yahoo] \
                        learner.samples_per_task=[10000,10000,1024,10000,1024,10000]


# python train_text_cls.py learner=anml name=test-mrelearning\
#                         task_order=[amazon] learner.samples_per_task=[32]\
#                         testing.eval_dataset=amazon testing.n_samples=32 testing.few_shot_batch_size=1
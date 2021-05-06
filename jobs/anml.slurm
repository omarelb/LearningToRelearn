#!/bin/bash

#SBATCH --job-name=anml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:10:00
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
python train_text_cls.py learner=anml name=test debug_logging=true wandb=false training=debug debug_data=false learner.samples_per_task=[64,64] task_order=[amazon,yahoo] testing.average_accuracy=false testing.n_samples=32


# python train_text_cls.py learner=anml name=test-mrelearning\
#                         task_order=[amazon] learner.samples_per_task=[32]\
#                         testing.eval_dataset=amazon testing.n_samples=32 testing.few_shot_batch_size=1
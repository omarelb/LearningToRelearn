#!/bin/bash

#SBATCH --job-name=measure
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

# Dimension -- How many samples in between:

# 1) t_1 (5000), t_2 (2000), t_1 (evaluate)

# python train_text_cls.py learner=sequential name=measure-relearning-baseline-1\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,2000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 2) t_1 (5000), t_2 (5000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-2\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,5000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 3) t_1 (5000), t_2 (10000), t_1 (evaluate)

# python train_text_cls.py learner=sequential name=measure-relearning-baseline-3\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 4) t_1 (5000), t_2 (50000), t_1 (evaluate)

# python train_text_cls.py learner=sequential name=measure-relearning-baseline-4\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,50000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 24) t_1 (5000), t_2 (20000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-24\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,20000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 25) t_1 (5000), t_2 (30000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-25\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,30000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# # Dimension -- How many samples of that task are seen the first time:
# 5) t_1 (2000), t_2 (5000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-5\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[2000,5000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 2) t_1 (5000), t_2 (5000), t_1 (evaluate)

# 6) t_1 (10000), t_2 (5000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-6\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[10000,5000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 7) t_1 (50000), t_2 (5000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-7\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,5000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 20) t_1 (50000), t_2 (10000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-20\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 21) t_1 (50000), t_2 (20000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-21\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,20000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 22) t_1 (50000), t_2 (50000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-22\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,50000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# # Dimension -- How many tasks are in between observing that task:
# 12) t_1 (5000), t_2 (30000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-12\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,30000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 13) t_1 (5000), t_2 (15000 (=30000/2)), t_3 (15000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-13\
#                         task_order=[amazon,yahoo,agnews] learner.samples_per_task=[5000,15000,15000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1
# 14) t_1 (5000), t_2 (10000 (=30000/3)), t_3 (10000), t_4 (10000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-14\
#                         task_order=[amazon,yahoo,agnews,dbpedia] learner.samples_per_task=[5000,10000,10000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 26) t_1 (50000), t_2 (30000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-26\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,30000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 27) t_1 (50000), t_2 (15000 (=30000/2)), t_3 (15000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-27\
#                         task_order=[amazon,yahoo,agnews] learner.samples_per_task=[50000,15000,15000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 28) t_1 (50000), t_2 (10000 (=30000/3)), t_3 (10000), t_4 (10000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-28\
#                         task_order=[amazon,yahoo,agnews,dbpedia] learner.samples_per_task=[50000,10000,10000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# # Dimension -- How many times we observe that task
# 2) t_1 (5000), t_2 (5000), t_1 (evaluate)
# 15) t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-15\
#                         task_order=[amazon,yahoo,amazon,yahoo] learner.samples_per_task=2500\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 10) t_1 (10000), t_2 (10000), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-10\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[10000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 16) t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (evaluate)
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-16\
#                         task_order=[amazon,yahoo,amazon,yahoo,amazon,yahoo,amazon,yahoo]\
#                         learner.samples_per_task=2500\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1


# Without observing the evaluated task
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-17\
#                         task_order=[yahoo] learner.samples_per_task=[10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# more few shot evaluation samples
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-18\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,50000]\
#                         testing.eval_dataset=amazon testing.n_samples=1024 testing.few_shot_batch_size=1

# python train_text_cls.py learner=sequential name=measure-relearning-baseline-19\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,50000]\
#                         testing.eval_dataset=amazon testing.n_samples=2048 testing.few_shot_batch_size=4

# zero shot
# python train_text_cls.py learner=sequential name=measure-relearning-baseline-23\
#                         task_order=[amazon] learner.samples_per_task=[0]\
#                         testing.eval_dataset=amazon testing.n_samples=2048 testing.few_shot_batch_size=1
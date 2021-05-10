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

# python train_text_cls.py learner=anml name=test-mrelearning\
#                         task_order=[amazon] learner.samples_per_task=[50]\
#                         testing.eval_dataset=amazon testing.n_samples=32 testing.few_shot_batch_size=1

# Dimension -- How many samples in between:

# 1) t_1 (5000), t_2 (2000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-1\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,2000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 2) t_1 (5000), t_2 (5000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-2\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,5000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 3) t_1 (5000), t_2 (10000), t_1 (evaluate)

# python train_text_cls.py learner=anml name=measure-relearning-anml-3\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 4) t_1 (5000), t_2 (50000), t_1 (evaluate)

# python train_text_cls.py learner=anml name=measure-relearning-anml-4\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,50000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 5) t_1 (5000), t_2 (20000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-5\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,20000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 6) t_1 (5000), t_2 (30000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-6\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,30000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# # Dimension -- How many samples of that task are seen the first time:
# 7) t_1 (2000), t_2 (5000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-7\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[2000,5000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 2) t_1 (5000), t_2 (5000), t_1 (evaluate)

# 8) t_1 (10000), t_2 (5000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-8\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[10000,5000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 9) t_1 (50000), t_2 (5000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-9\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,5000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 10) t_1 (50000), t_2 (10000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-10\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 11) t_1 (50000), t_2 (20000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-11\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,20000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 12) t_1 (50000), t_2 (50000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-12\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,50000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# # Dimension -- How many tasks are in between observing that task:
# 13) t_1 (5000), t_2 (30000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-13\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,30000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 14) t_1 (5000), t_2 (15000 (=30000/2)), t_3 (15000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-14\
#                         task_order=[amazon,yahoo,agnews] learner.samples_per_task=[5000,15000,15000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 15) t_1 (5000), t_2 (10000 (=30000/3)), t_3 (10000), t_4 (10000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-15\
#                         task_order=[amazon,yahoo,agnews,dbpedia] learner.samples_per_task=[5000,10000,10000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 16) t_1 (50000), t_2 (30000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-16\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,30000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 17) t_1 (50000), t_2 (15000 (=30000/2)), t_3 (15000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-17\
#                         task_order=[amazon,yahoo,agnews] learner.samples_per_task=[50000,15000,15000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 18) t_1 (50000), t_2 (10000 (=30000/3)), t_3 (10000), t_4 (10000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-18\
#                         task_order=[amazon,yahoo,agnews,dbpedia] learner.samples_per_task=[50000,10000,10000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 26) t_1 (50000), t_2 (100000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-1024-26\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[50000,100000]\
#                         testing.eval_dataset=amazon testing.n_samples=1024 testing.few_shot_batch_size=1

# 27) t_1 (50000), t_2 (50000 (=100000/2)), t_3 (50000), t_1 (evaluate)
python train_text_cls.py learner=anml name=measure-relearning-anml-1024-27\
                        task_order=[amazon,yahoo,agnews] learner.samples_per_task=[50000,50000,50000]\
                        testing.eval_dataset=amazon testing.n_samples=1024 testing.few_shot_batch_size=1

# 28) t_1 (50000), t_2 (33333 (=100000/3)), t_3 (33333), t_4 (33333), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-1024-28\
#                         task_order=[amazon,yahoo,agnews,dbpedia] learner.samples_per_task=[50000,33000,33000,33000]\
#                         testing.eval_dataset=amazon testing.n_samples=1024 testing.few_shot_batch_size=1

# # Dimension -- How many times we observe that task
# 2) t_1 (5000), t_2 (5000), t_1 (evaluate)
# 19) t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-19\
#                         task_order=[amazon,yahoo,amazon,yahoo] learner.samples_per_task=2500\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 20) t_1 (10000), t_2 (10000), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-20\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[10000,10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# 21) t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (evaluate)
# python train_text_cls.py learner=anml name=measure-relearning-anml-21\
#                         task_order=[amazon,yahoo,amazon,yahoo,amazon,yahoo,amazon,yahoo]\
#                         learner.samples_per_task=2500\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1


# 22) Without observing the evaluated task
# python train_text_cls.py learner=anml name=measure-relearning-anml-22\
#                         task_order=[yahoo] learner.samples_per_task=[10000]\
#                         testing.eval_dataset=amazon testing.n_samples=256 testing.few_shot_batch_size=1

# more few shot evaluation samples
# 23)
# python train_text_cls.py learner=anml name=measure-relearning-anml-23\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,50000]\
#                         testing.eval_dataset=amazon testing.n_samples=1024 testing.few_shot_batch_size=1

# 24)
# python train_text_cls.py learner=anml name=measure-relearning-anml-24\
#                         task_order=[amazon,yahoo] learner.samples_per_task=[5000,50000]\
#                         testing.eval_dataset=amazon testing.n_samples=2048 testing.few_shot_batch_size=4

# zero shot
# 25)
# python train_text_cls.py learner=anml name=measure-relearning-anml-25\
#                         task_order=[amazon] learner.samples_per_task=[0]\
#                         testing.eval_dataset=amazon testing.n_samples=2048 testing.few_shot_batch_size=1

# 29) A whole lot of samples
# python train_text_cls.py learner=anml name=measure-relearning-anml-1024-29\
#                         task_order=[amazon,yahoo,agnews,dbpedia] learner.samples_per_task=[50000,115000,115000,115000]\
#                         testing.eval_dataset=amazon testing.n_samples=1024 testing.few_shot_batch_size=1
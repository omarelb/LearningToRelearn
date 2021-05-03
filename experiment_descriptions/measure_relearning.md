# Measure relearning

## Goal
 
We want to learn to what extent models can use their knowledge of a previously seen task to learn that task or a similar task more quickly. Simultaneously, we want to measure how much the learners are forgetting after not having seen a task for a while.

We want to do this for multiple different learners and different orders in which data is presented to that learner.

## Setup
Multiple models are tested, of which at least:

- Vanilla BERT (baseline)
- ANML, with and without replay
- Own model (which is still in development)

The order in which data is presented to the model is important, and will/can change along the following dimensions:
- Which task is evaluated
- How many samples of that task are seen the first time
- How many samples of that task are seen the following times
- How many samples are in between observing that task
- How many tasks are in between observing that task
- How many times we observe that task
- How many samples are seen in total
- The order of tasks observed in between


We test the following data orders:
Format: Task (n_samples).
If the item in the numbered list has the same number as a previous one, the same run can be used.

Dimension -- How many samples in between:
1) t_1 (5000), t_2 (2000), t_1 (evaluate)
2) t_1 (5000), t_2 (5000), t_1 (evaluate)
3) t_1 (5000), t_2 (10000), t_1 (evaluate)
4) t_1 (5000), t_2 (50000), t_1 (evaluate)

Dimension -- How many samples of that task are seen the first time:
5) t_1 (2000), t_2 (5000), t_1 (evaluate)
2) t_1 (5000), t_2 (5000), t_1 (evaluate)
6) t_1 (10000), t_2 (5000), t_1 (evaluate)
7) t_1 (50000), t_2 (5000), t_1 (evaluate)
The 4 below maybe
8) t_1 (2000), t_2 (10000), t_1 (evaluate)
3) t_1 (5000), t_2 (10000), t_1 (evaluate)
10) t_1 (10000), t_2 (10000), t_1 (evaluate)
11) t_1 (50000), t_2 (10000), t_1 (evaluate)

Dimension -- How many tasks are in between observing that task:
12) t_1 (5000), t_2 (30000), t_1 (evaluate)
13) t_1 (5000), t_2 (15000 (=30000/2)), t_3 (15000), t_1 (evaluate)
14) t_1 (5000), t_2 (10000 (=30000/3)), t_3 (10000), t_4 (10000), t_1 (evaluate)

Dimension -- How many times we observe that task
2) t_1 (5000), t_2 (5000), t_1 (evaluate)
15) t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (evaluate)
10) t_1 (10000), t_2 (10000), t_1 (evaluate)
16) t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (evaluate)

We then need to evaluate the models. We do this as follows:
- We measure the accuracy during training, and call this the online accuracy.
- We use the area under the performance curve to measure relearning speed. We record zero shot
    performance, and the area under the performance curve minus the zero shot performance gives us a measure of relearning.
- During evaluation, we train the learner on a small amount of samples at a time, and evaluate on a held out dataset, which gives us our performance curve.


## Further Info
*slurm jobs located in*: `jobs/measure_relearning`

*run name*: `measure-relearning-{learner_name}`
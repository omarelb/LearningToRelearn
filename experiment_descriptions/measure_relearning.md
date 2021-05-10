# Measure relearning

## Goal
 
We want to learn to what extent models can use their knowledge of a previously seen task to learn that task or a similar task more quickly. Simultaneously, we want to measure how much the learners are forgetting after not having seen a task for a while.

We want to do this for multiple different learners and different orders in which data is presented to that learner.

## Setup
Multiple models are tested, of which at least:

- Vanilla BERT (baseline, sequential)
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
t_1 (5000), t_2 (2000), t_1 (evaluate)
t_1 (5000), t_2 (5000), t_1 (evaluate)
t_1 (5000), t_2 (10000), t_1 (evaluate)
t_1 (5000), t_2 (50000), t_1 (evaluate)
t_1 (5000), t_2 (20000), t_1 (evaluate)
t_1 (5000), t_2 (30000), t_1 (evaluate)

t_1 (50000), t_2 (10000), t_1 (evaluate)
t_1 (50000), t_2 (20000), t_1 (evaluate)
t_1 (50000), t_2 (50000), t_1 (evaluate)

Dimension -- How many samples of that task are seen the first time:
t_1 (2000), t_2 (5000), t_1 (evaluate)
t_1 (5000), t_2 (5000), t_1 (evaluate)
t_1 (10000), t_2 (5000), t_1 (evaluate)
t_1 (50000), t_2 (5000), t_1 (evaluate)

Dimension -- How many tasks are in between observing that task:
t_1 (5000), t_2 (30000), t_1 (evaluate)
t_1 (5000), t_2 (15000 (=30000/2)), t_3 (15000), t_1 (evaluate)
t_1 (5000), t_2 (10000 (=30000/3)), t_3 (10000), t_4 (10000), t_1 (evaluate)

t_1 (50000), t_2 (30000), t_1 (evaluate)
t_1 (50000), t_2 (15000 (=30000/2)), t_3 (15000), t_1 (evaluate)
t_1 (50000), t_2 (10000 (=30000/3)), t_3 (10000), t_4 (10000), t_1 (evaluate)

t_1 (50000), t_2 (100000), t_1 (evaluate)
t_1 (50000), t_2 (50000 (=100000/2)), t_3 (50000), t_1 (evaluate)
t_1 (50000), t_2 (33333 (=100000/3)), t_3 (33333), t_4 (33333), t_1 (evaluate)

Dimension -- How many times we observe that task
t_1 (5000), t_2 (5000), t_1 (evaluate)
t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (evaluate)
t_1 (10000), t_2 (10000), t_1 (evaluate)
t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (2500), t_2 (2500), t_1 (evaluate)


We then need to evaluate the models. We do this as follows:
- We measure the accuracy during training, and call this the online accuracy.
- We use the area under the performance curve to measure relearning speed. We record zero shot
    performance, and the area under the performance curve minus the zero shot performance gives us a measure of relearning.
    The zero-shot performance itself indicates how much the model remembers instead of relearns. A higher zero-shot performance
    indicates a lower amount of forgetting.
- During evaluation, we train the learner on a small amount of samples at a time, and evaluate on a held out dataset, which gives us our performance curve.

Finally, we compare different models on the same orders and different orders on the same models.

When comparing models, we report a table containing the following, where metrics are aggregated over all data orders or a specific order is chosen.
| model name | lca@0 | lca_difference@16 | lca_difference@64 | lca_difference@128 | lca_difference@256 | lca_difference@512 | lca_difference@1024|

Qualitatively, we can plot the few shot learning curve for a single order for multiple models.

To compare different orders, we plot the learning curve areas of the same model

## Further Info
*slurm jobs located in*: `jobs/measure_relearning`

*run name*: `measure-relearning-{learner_name}`

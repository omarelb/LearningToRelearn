# Single vs Multitask

## Goal
We want to compare the performance of a model trained on tasks individually, to that
of a model trained on all those tasks simultaneously.

We hypothesize that it is difficult for a single model to be good at many tasks simultaneously, and
that this becomes more apparent the more tasks need to be learned simultaneously. Possible reasons
would be a limit in model capacity, or interference between tasks.

If this is true, it shows the necessity of using model capacity efficiently, and that we might not
want to keep everything in memory at the same time.

## Setup
1) Train models on single tasks and evaluate on same tasks
2) Train models on multiple tasks simultaneously, where we increase the number of tasks trained on
each time.
3) Compare performances.

## Further Info
*slurm jobs located in*: `jobs/single_vs_multitask`

*run name*: `single-vs-multitask`
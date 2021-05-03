# Amazon vs Yelp

## Description
Amazon and Yelp tasks are both sentiment classification tasks with 5 degrees of sentiment.
I have observed that performance on one task transfers to the other without the need to train on
the other task. Therefore one of the tasks should be removed when evaluating continual learning, as what
seems like two different tasks really is just one task.

## Goal
Show that Yelp and Amazon are essentially the same tasks. Therefore, we should only be using
one of them in our CL evaluations.

## Setup
1) Train on Amazon
2) Evaluate on Yelp

3) Train on Yelp
4) Evaluate on Yelp

If difference between the two Yelp evaluations is small, we see that training on Amazon is
basically equivalent to training on Yelp.

## Further Info
*slurm job located in*: `jobs/amazon_vs_yelp.slurm`

*run name*: `amazon_vs_yelp`
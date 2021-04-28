Models
---

Every model in this folder defines a learning method, inheriting from a base learner that does all the bookkeeping.

Each model implements its own training loop, testing loop, and evaluation method (which is used during e.g. validation and in the testing loop.)

Each learner should implement the following methods:
- training: training loop
- testing
- set_eval: setting all model parts to eval mode. For learners with only a single model, this just means doing self.model.eval(). Learners that only do this don't have to implement this method.

Each model should do the following in their training loop:
- adding training accuracy to the metrics at every iteration, using 
    ```self.metrics["online"].append(acc)```
- increment the `current_iter` counter by 1
- validating using `self.validate`
- write to weights and biases (wandb) if specified in config
- increment `examples_seen` if not implemented the function `examples_seen`
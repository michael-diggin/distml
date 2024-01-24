# distml
Multi worker distributed deep learning from scratch

As part of getting more exposure to MLOps and ML Engineering, this is a small
project to implement a very basic multi worker distributed training set up.

It's based on and supports models made using tf.keras, the interface aims to resemble it, with a similar `model.fit` method. Data is handled using tf.Datasets which are pretty powerful, even when only the basics are used.

It's essentially a synchronous all-reduce distributed set up.
One node is the 'leader' and pushes weights to each worker node. They all run a single training step
on a sharded batch of input data, and return the gradients to the leader.
The leader accumulates those gradients, performs the update step, and repeats.
It can handle some distributed failures (e.g node restarts) fairly well.

A simple mnist example can be run locally:
- `NODE=2 PORT=1232 python3 mnist_example.py` in one terminal window
- `NODE=1 PORT=1231 python3 mnist_example.py` in another terminal window
- `NODE=0 PORT=1230 python3 mnist_example.py` in another terminal window

Following the logs in the last terminal will output the loss for each epoch and finish by
printing out the accuracy on a portion of the test set.


### to do (no particular order):
- implement metrics aggregation
- implement checkpointing for optimizer
- implement losses visualisation
- add bi-directional streaming for larger data transfers
- add in validation inference and early stopping
- support for regularization
- use proper logging
- add checkpointing to GCS
- set up k8s manifests and run on GKE for some dataset
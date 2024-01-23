# distml
Multi worker distributed deep learning from scratch

As part of getting more exposure to MLOps and ML Engineering, this is a small
project to implement a very basic multi worker distributed training set up.

It's based on and supports models made using tf.keras, the interface aims to resemble it, with a similar `model.fit` method. Data is handled using tf.Datasets which are pretty powerful, even when only
the basics are used.


### Implementation to do (no particular order):
- handle main training loop when replicas are down, and set gRPC timeouts
- implement metrics aggregation
- implement checkpointing for optimizer
- implement losses visualisation
- add bi-directional streaming for larger data transfers
- add in validation inference and early stopping
- support for regularization
- use proper logging
- add checkpointing to GCS
- set up k8s manifests and run on GKE for some dataset
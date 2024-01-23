# distml
Multi worker distributed deep learning from scratch


## Implementation to do:

- implement get_next_batch when input is a tf.Data type
- ensure tf functions and tensors are being used correctly
- handle main training loop when replicas are down
- implement metrics aggregation
- implement checkpointing for optimizer
- add checkpointing to GCS
- implement losses visualisation
- Move to bi-directional streaming for larger data transfers
- Add in validation inference and early stopping
- Support for regularization
- Use proper logging
- set up k8s manifests and run on GKE for some dataset
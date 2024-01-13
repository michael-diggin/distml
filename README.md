# distml
Multi worker distributed deep learning from scratch


## Implementation to do:

- implement get_next_batch function
- implement run_train_step
- implement update_params
- implement fit epoch/step loop
- implement metrics aggregation
- implement checkpointing (weights and epoch)
- wait for workers to be available on startup

- Move to bi-directional streaming for larger data transfers

For 'main', flow is:
- Stream weights to learning nodes
- run a batch of learning, compute grads
- receive grads from learning nodes
- update weights given the grads
- report metrics

For 'learning' nodes:
- Receive weights from main node
- Run batch of learning, compute grads
- Stream grads to main node

Running a batch of learning includes:
- Getting batch of data, and sharding it in each node
- Feedforward through the ml model to get predictions
- Compute loss and gradients
- Report local metrics
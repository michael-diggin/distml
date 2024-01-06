# distml
Multi worker distributed deep learning from scratch


## Implementation to do:
- Define proto -> start with request/response (using futures), move to streaming later
for larger weights/grad sizes
- Interface needed is RunStep(weights) returns Grads
- create server and client objects
- Create DistModel class -> wraps the server, client, ml model
- Serialising lists of numpy arrays and TF eager tensors
- Simple case first, handle failure events later

Weights/grads are coupled with epoch number and batch number


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
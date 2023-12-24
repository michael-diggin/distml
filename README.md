# distml
Multi worker distributed deep learning from scratch


## Implementation to do:
- Define proto -> check if bidirectional streaming is best option?
- Interface needed is SetWeights and SyncGrads
- create server and client objects
- Create DistModel class -> wraps the server, client, ml model
- Simple case first, handle failure events later

Weights/grads are coupled with epoch number and bacth number


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
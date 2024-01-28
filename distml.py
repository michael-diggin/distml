import train_pb2_grpc
import train_pb2
from concurrent import futures
import grpc
import math
import numpy as np
import tensorflow as tf
import serialize
from retry import retry_on_statuscode
from checkpoint import NoopCheckpoint


class TrainerServer(train_pb2_grpc.TrainerServicer):
    def __init__(self, dist_config, model, loss_fn, optimizer, checkpointer=NoopCheckpoint()):
        self.conf = dist_config
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.epoch = -1
        self.step = -1
        self.current_batch = None
        self.batch_split = 1 + len(dist_config["servers"]) # 1 leader and other servers
        self.node_ix = dist_config["node"]
        self.node_int = int(self.node_ix)
        self.server = None
        self._channels = {}
        self.workers = [] # list of other server worker addresses
        self.worker_stubs = {}
        self.grpc_timeout = 60 #seconds
        if dist_config["leader"] == dist_config["node"]:
            self._setup_leader(dist_config)
        else:
            self._setup_server(dist_config["port"])

    def _setup_server(self, port):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        train_pb2_grpc.add_TrainerServicer_to_server(
            self, server
        )
        addr = f"[::]:{port}"
        server.add_insecure_port(addr)
        self.server = server

    def _setup_leader(self, conf):
        # set up stubs
        self.workers = conf["servers"]
        for addr in self.workers:
            self._connect_to_node(addr)
    
    @retry_on_statuscode(0, 5, [grpc.StatusCode.UNAVAILABLE])
    def _connect_to_node(self, addr):
        existing_stub = self.worker_stubs.get(addr, None)
        if existing_stub:
            try:
                existing_stub.HeartBeat(train_pb2.HeartBeatRequest(), timeout=5)
                # stub already exists and works, continue
                return
            except:
                # connection is faulty, close it's channel and create a new one
                self._channels[addr].close()
                pass

        channel = grpc.insecure_channel(addr)
        stub = train_pb2_grpc.TrainerStub(channel)
        # establish a connection here
        stub.HeartBeat(train_pb2.HeartBeatRequest(), timeout=5)
        self.worker_stubs[addr] = stub
        self._channels[addr] = channel
        return
        
    def close(self):
        for channel in self._channels.values():
            channel.close()
        if self.server:
            self.server.stop(grace=5)

    def fit(self, epochs, batch_size=None, x_data=None, y_data=None, dataset=None, num_steps=None, validation_dataset=None, validation_steps=None):
        assert (x_data is not None or dataset), "At least one of dataset or (x_data, y_data) must be set"
        if x_data is not None:
            # creates a dataset object from the input tensors/arrays, sets the batch and the shard
            dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size).shard(self.batch_split, self.node_int)
        # otherwise assume that it's batched and sharded?
        self.dataset = dataset
        self.data_iter = iter(self.dataset)
        self.val_dataset = validation_dataset
        self.val_steps = validation_steps
        if self.server:
            self.server.start()
            self.server.wait_for_termination()
            return
        # otherwise

        last_epoch = self._load_latest_checkpoint()
        self.epoch = last_epoch if last_epoch else -1

        print("-------------- Starting Training --------------")
        if not num_steps:
            num_steps = math.floor(x_data.shape[0] / (self.batch_split*batch_size))
        losses = []
        for epoch in range(last_epoch+1, epochs):
            epoch_losses = []

            for step in range(num_steps):
                gradients, step_losses = self._run_distributed_step(epoch, step)
                self._update_with_grads(gradients)
                losses.append(np.mean(step_losses))
                epoch_losses.append(np.mean(step_losses))
            
            output = f"Epoch {epoch}: {np.mean(epoch_losses)}"
            
            if self.checkpointer.should_checkpoint(epoch):
                opt_state = {}
                self.optimizer.save_own_variables(opt_state)
                self.checkpointer.save_weights(epoch, self.model.get_weights(), opt_state)

            # run some validation if val_dataset is exists
            if self.val_dataset:
                val_loss = self._run_distributed_validation(epoch, self.model.get_weights())
                output += f"\tValidation loss = {val_loss}"
            print(output)

        print("Finished Training")
        for addr in self.workers:
            self.worker_stubs[addr].Finish(train_pb2.FinishRequest())
        self.close()

    def _load_latest_checkpoint(self):
        last_epoch, weights, opt_weights = self.checkpointer.load_latest_weights()
        if weights:
            if opt_weights:
                # need to show the optimizer what variable sizes it needs to have
                # and set model weights afterwards
                zero_grads = [tf.zeros_like(v) for v in self.model.trainable_variables]
                self.optimizer.apply_gradients(zip(zero_grads, self.model.trainable_variables))
                opt_state = {str(i): opt_weights[i] for i in range(len(opt_weights))}
                self.optimizer.load_own_variables(opt_state)
            self.model.set_weights(weights)
            return last_epoch
        return None

    @retry_on_statuscode(0, 5, [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED])
    def _run_distributed_step(self, epoch, step):
        try:
            futures = []
            weights = serialize.weights_to_proto(self.model.get_weights())
            req = train_pb2.RunStepReuest(epoch=epoch, step=step, weights=weights)
            for addr in self.workers:
                futures.append(self._send_train_request(addr, req))

            loss, grads = self._run_model_train_step(epoch, step)
            step_losses = [loss]
            gradients = [grads]
            for f in futures:
                resp = f.result()
                f_grads = serialize.grads_from_proto(resp.grads)
                f_loss = serialize.loss_from_proto(resp.loss)
                step_losses.append(f_loss)
                gradients.append(f_grads)
            return gradients, step_losses
        except grpc.RpcError as grpc_error:
            if grpc_error.code() in [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED]:
                # refresh all unhealthy connections
                self._setup_leader(self.conf)
                raise grpc_error
    
    def _send_train_request(self, addr, req):
        stub = self.worker_stubs[addr]
        resp_future = stub.RunStep.future(req, timeout=self.grpc_timeout)
        return resp_future
    
    def _send_val_request(self, addr, req):
        stub = self.worker_stubs[addr]
        resp_future = stub.RunValidation.future(req, timeout=120)
        return resp_future

    def RunStep(self, request, context):
        weights = serialize.weights_from_proto(request.weights)
        self.model.set_weights(weights)
        loss, grads = self._run_model_train_step(request.epoch, request.step)
        grads = serialize.grads_to_proto(grads)
        loss = serialize.loss_to_proto(loss)
        resp = train_pb2.RunStepResponse(epoch=request.epoch, step=request.step, loss=loss, grads=grads)
        return resp
    
    def RunValidation(self, request, context):
        weights = serialize.weights_from_proto(request.weights)
        self.model.set_weights(weights)
        num_steps = request.num_steps
        loss = self._run_validation(num_steps)
        resp = train_pb2.RunValidationResponse(loss=serialize.loss_to_proto(loss))
        return resp
    
    def _run_validation(self, num_steps):
        val_iter = iter(self.val_dataset)
        losses = []
        for _ in range(num_steps):
            x, y = next(val_iter)
            preds = self.model(x)
            losses.append(self.loss_fn(y, preds))
        loss = tf.add_n(losses)
        return loss

    def HeartBeat(self, request, context):
        return train_pb2.HeartBeatResponse()
    
    def Finish(self, request, context):
        print("Received finish request, shutting down")
        self.close()
        return train_pb2.FinishResponse()

    def _run_model_train_step(self, epoch, step):
        x, y = self._get_batch(epoch, step)
        with tf.GradientTape() as tape:
            preds = self.model(x)
            loss = self.loss_fn(y, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, grads

    # using tf.Dataset allows for not needing to worry about sharding the data batch here.
    def _get_batch(self, epoch, step):
        if (epoch == self.epoch and step == self.step and self.current_batch is not None):
            # small optimisation
            # in case a node restarted and a training step gets retried
            return self.current_batch
        if (epoch == self.epoch and step == self.step +1):
            x, y = next(self.data_iter)
            self.step = step
            self.current_batch = (x, y)
            return x, y
        else:
            self.epoch = epoch
            self.step = step
            # two possibilities
            # 1. new epoch, first batch should be returned.
            # 2. worker node got restarted, iterate to find the right batch
            #   not very effecient but it is what it is (for now).
            self.data_iter = iter(self.dataset)
            x, y = next(self.data_iter)
            for _ in range(1, step+1):
                x, y = next(self.data_iter)
        self.current_batch = (x, y)
        return x, y

    def _update_with_grads(self, grads):
        # combine the N sets of grads
        # has to be a better way to accumulate them
        gradients = grads[0]
        for i in range(len(grads[0])):
            for g in range(1, len(grads)):
                gradients[i] = tf.math.add(gradients[i], grads[g][i])
            gradients[i] = gradients[i] / self.batch_split
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return

    def _run_distributed_validation(self, epoch, weights):
        futures = []
        w = serialize.weights_to_proto(weights)
        req = train_pb2.RunValidationRequest(epoch=epoch, num_steps=self.val_steps, weights=w)
        for addr in self.workers:
            futures.append(self._send_val_request(addr, req))
        losses = [self._run_validation(self.val_steps)]
        
        for f in futures:
            resp = f.result()
            losses.append(serialize.loss_from_proto(resp.loss))
        loss = tf.add_n(losses).numpy() / (self.batch_split*self.val_steps)
        return loss
        
        


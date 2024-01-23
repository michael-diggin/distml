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
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.epoch = 0
        self.step = -1
        self.batch_split = 1 + len(dist_config["servers"]) # 1 leader and other servers
        self.node_ix = dist_config["node"]
        self.node_int = int(self.node_ix)
        self.server = None
        self._channels = []
        self.workers = [] # list of other server worker addresses
        self.worker_stubs = {}
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
        channel = grpc.insecure_channel(addr)
        stub = train_pb2_grpc.TrainerStub(channel)
        # establish a connection here
        stub.HeartBeat(train_pb2.HeartBeatRequest())
        self.worker_stubs[addr] = stub
        self._channels.append(channel)
        return
        
    def close(self):
        for channel in self._channels:
            channel.close()
        if self.server:
            self.server.stop(grace=5)

    def fit(self, epochs, batch_size=None, x_data=None, y_data=None, dataset=None, num_steps=None):
        assert (x_data is not None or dataset), "At least one of dataset or (x_data, y_data) must be set"
        if x_data is not None:
            # creates a dataset object from the input tensors/arrays, sets the batch and the shard
            dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).batch(batch_size).shard(self.batch_split, self.node_int)
        # otherwise assume that it's batched and sharded?
        self.dataset = dataset
        self.data_iter = iter(self.dataset)
        if self.server:
            self.server.start()
            self.server.wait_for_termination()
            return
        # otherwise

        last_epoch, weights = self.checkpointer.load_latest_weights()
        if weights:
            self.model.set_weights(weights)
        self.epoch = last_epoch

        print("-------------- Starting Training --------------")
        if not num_steps:
            num_steps = math.floor(x_data.shape[0] / (self.batch_split*batch_size))
        losses = []
        for epoch in range(last_epoch, epochs):
            epoch_losses = []

            for step in range(num_steps):
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

                self._update_with_grads(gradients)
                losses.append(np.mean(step_losses))
                epoch_losses.append(np.mean(step_losses))
            
            if epoch%1 == 0:
                print(f"Epoch {epoch}: {np.mean(epoch_losses)}")
            
            if self.checkpointer.should_checkpoint(epoch):
                self.checkpointer.save_weights(epoch, self.model.get_weights())

        print("Finished Training")
        for addr in self.workers:
            self.worker_stubs[addr].Finish(train_pb2.FinishRequest())
        self.close()

    
    def _send_train_request(self, addr, req):
        stub = self.worker_stubs[addr]
        resp_future = stub.RunStep.future(req)
        return resp_future


    def RunStep(self, request, context):
        weights = serialize.weights_from_proto(request.weights)
        self.model.set_weights(weights)
        loss, grads = self._run_model_train_step(request.epoch, request.step)
        grads = serialize.grads_to_proto(grads)
        loss = serialize.loss_to_proto(loss)
        resp = train_pb2.RunStepResponse(epoch=request.epoch, step=request.step, loss=loss, grads=grads)
        return resp
    
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

    # this method isn't needed since everything is now wrapped up into
    # tf.Datasets, which handle the batch and sharding automatically.
    # def _get_batch_from_data(self, epoch, step):
        # TODO: this can be removed by created a tf Dataset from the arrays
        # once they are passed to fit
        # grabs the next batch_size of data
    #    self.step = step
    #    start = self.step*self.batch_size
    #    end = min((self.step+1)*self.batch_size, self.x_data.shape[0])
    #    x_data = self.x_data[start:end,:]
    #    y_data = self.y_data[start:end,:]

        # splits it equally amongst the nodes
        # if the data to split is < number of splits
        # give all workers that small amount of data
    #    length = end-start
    #    if length < self.batch_split:
    #        return x_data, y_data
    #    split = [i for i in range(0, length, length//self.batch_split)] + [length]
    #    x_batch = x_data[split[self.node_int]:split[self.node_int+1], :]
    #    y_batch = y_data[split[self.node_int]:split[self.node_int+1], :]
    #    return x_batch, y_batch
    
    def _get_batch(self, epoch, step):
        if (epoch == self.epoch and step == self.step +1):
            x, y = next(self.data_iter)
            self.step = step
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



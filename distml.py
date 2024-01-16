import train_pb2_grpc
import train_pb2
from concurrent import futures
import grpc
import time
import numpy as np
import tensorflow as tf
import serialize
from retry import retry_on_statuscode


class TrainerServer(train_pb2_grpc.TrainerServicer):
    def __init__(self, dist_config, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epoch = 0
        self.batch_number = 0
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

    def fit(self, epochs, batch_size, x_data, y_data):
        self.batch_size = batch_size
        self.x_data = x_data
        self.y_data = y_data
        if self.server:
            self.server.start()
            self.server.wait_for_termination()
            return
        # otherwise
        print("-------------- Starting Training --------------")
        step = 0
        for epoch in range(epochs):
            futures = []
            weights = serialize.weights_to_proto(self.model.get_weights())
            req = train_pb2.RunStepReuest(epoch=epoch, step=step, weights=weights)
            for addr in self.workers:
                futures.append(self._send_train_request(addr, req))
            
            loss, grads = self._run_model_train_step(epoch, step)
            losses = [loss]
            gradients = [grads]
            for f in futures:
                resp = f.result()
                f_grads = serialize.grads_from_proto(resp.grads)
                f_loss = serialize.loss_from_proto(resp.loss)
                losses.append(f_loss)
                gradients.append(f_grads)

            epoch_loss = np.mean(losses)
            print(f"Epoch Loss: {epoch_loss}")

            self._update_with_grads(gradients)

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
        self.epoch = request.epoch
        resp = train_pb2.RunStepResponse(epoch=request.epoch, step=request.step, loss=loss, grads=grads)
        return resp
    
    def HeartBeat(self, request, context):
        return train_pb2.HeartBeatResponse()
    
    def Finish(self, request, context):
        print("Received finish request, shutting down")
        self.close()
        return train_pb2.FinishResponse()

    def _run_model_train_step(self, epoch, batch_number):
        x, y = self._get_batch(epoch, batch_number)
        with tf.GradientTape() as tape:
            preds = self.model(x)
            loss = self.loss_fn(y, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)
        return loss, grads

    def _get_batch(self, epoch, step):
        # for now - just all the data
        length = self.x_data.shape[0]
        split = [i for i in range(0, length, length//self.batch_split)] + [length]
        x_batch = self.x_data[split[self.node_int]:split[self.node_int+1], :]
        y_batch = self.y_data[split[self.node_int]:split[self.node_int+1], :]
        return x_batch, y_batch

    def _update_with_grads(self, grads):
        # combine the N sets of grads
        # has to be a better way to accumulate them
        gradients = grads[0]
        for i in range(len(grads[0])):
            for g in range(1, len(grads)):
                gradients[i] += grads[g][i]
            gradients[i] /= len(grads)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return



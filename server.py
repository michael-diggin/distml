import train_pb2_grpc
import train_pb2
from concurrent import futures
import grpc
import os
import time
import numpy as np
import tensorflow as tf
import serialize

class TrainerServer(train_pb2_grpc.TrainerServicer):
    def __init__(self, dist_config, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epoch = 0
        self.batch_number = 0
        self.batch_split = 1 + len(dist_config["servers"]) # 1 leader and other servers
        self.node_ix = conf["node"]
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
    
    # needs a retry decorator on status.UNAVAILABLE
    def _connect_to_node(self, addr):
        channel = grpc.insecure_channel(addr)
        stub = train_pb2_grpc.TrainerStub(channel)
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
        for epoch in range(epochs):
            futures = []
            # self.model.get_weights() <- used to get params to send to other nodes
            weights = [np.random.randn(2, 2)]
            weights = serialize.weights_to_proto(weights)
            req = train_pb2.RunStepReuest(epoch=epoch, step=0, weights=weights)
            for addr in self.workers:
                futures.append(self._send_train_request(addr, req))
        
            # would process on the leader's node as well
            # now collect results from workers
            print(f"Node {self.node_ix} handled task")
            for f in futures:
                resp = f.result()
                grads = serialize.grads_from_proto(resp.grads)
                print(grads[0])
            time.sleep(1)

        print("Finished Training")
        for addr in self.workers:
            self.worker_stubs[addr].Finish(train_pb2.FinishRequest())

        self.close()

    
    def _send_train_request(self, addr, req):
        stub = self.worker_stubs[addr]
        resp_future = stub.RunStep.future(req)
        return resp_future


    def RunStep(self, request, context):
        # this is where the ML model would run a step
        # weights = request.data
        # model.set_weights(weights)
        # get next batch
        # get preds, loss, grads
        # return response
        print("Running a Training Step!")
        weights = serialize.weights_from_proto(request.weights)
        print(weights)
        grads = tf.constant(weights)
        grads = serialize.grads_to_proto(grads)
        self.epoch = request.epoch
        resp = train_pb2.RunStepResponse(epoch=request.epoch, step=request.step, grads=grads)
        return resp
    
    def Finish(self, request, context):
        print("Received finish request, shutting down")
        self.close()
        return train_pb2.FinishResponse()

    def _run_model_train_step(self, epoch, batch_number):
        # get next batch of data
        # run forward pass
        # y_pred = self.model(x_batch)
        # get loss
        # l = self.loss_fn(y_pred, y_batch)
        # get grads
        # grads = tape.gradients(self.model.trainable_varaibles)
        # return l, grads
        pass

    def _get_batch(self, epoch, step):
        # if data is not an iterator, just pull out the next
        # grouping
        # otherwise iterate with next()
        # if epoch of step is different to what is expected
        # trust it and override local. Should only happen in a failure event
        # then split it with node_ix
        pass

    def _update_with_grads(self, grads):
        # combine the N sets of grads
        # self.optimizer.apply_gradients(zip(grads, model_vars))
        pass




if __name__ == "__main__":
    conf = {
        "leader": "0",
        "servers": ["localhost:1234"]#, "localhost:1235"],
    }
    node_ix = os.environ.get("NODE", "")
    port = os.environ.get("PORT", 1234)
    conf["node"] = node_ix
    conf["port"] = port

    print(f"Serving on {port}")
    ts = TrainerServer(conf, None, None, None)
    ts.fit(1, None, None, None)


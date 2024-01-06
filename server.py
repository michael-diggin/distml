import train_pb2_grpc
import train_pb2
from concurrent import futures
import grpc
import os

"""
config looks like
leader: node-name
servers: addr1, addr2, addr3, ...
"""




class TrainerServer(train_pb2_grpc.TrainerServicer):
    def __init__(self, dist_config):
        self.server = None
        self.workers = [] # list of other server worker addresses
        self.worker_stubs = {}
        if dist_config["leader"] == dist_config["node"]:
            self._setup_leader(dist_config)
        else:
            self._setup_server(dist_config["port"])

    def fit(self):
        if self.server:
            self.server.start()
            self.server.wait_for_termination()
        # otherwise
        print("-------------- RunTrainStep --------------")
        futures = []
        input = "Is it working?"
        data = str.encode(input)
        req = train_pb2.RunStepReuest(epoch=0, step=1, data=data)
        for addr in self.workers:
            futures.append(self._send_request(addr, req))
        
        # would process on the leader's node as well
        print("Handling my own task!")

        # now collect results from workers
        for f in futures:
            resp = f.result()
            print(resp.grads.decode())

    
    def _send_request(self, addr, req):
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
        req_data = request.data.decode()
        if req_data == "Is it working?":
            output_data = str.encode("It's working!!")
        resp = train_pb2.RunStepResponse(epoch=request.epoch, step=request.step, grads=output_data)
        return resp

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
            channel = grpc.insecure_channel(addr)
            # here we should just wait and keep retrying until we can connect
            stub = train_pb2_grpc.TrainerStub(channel)
            self.worker_stubs[addr] = stub




if __name__ == "__main__":
    conf = {
        "leader": "0",
        "servers": ["localhost:1234", "localhost:1235"],
    }
    node_ix = os.environ.get("NODE", "")
    port = os.environ.get("PORT", 1234)
    conf["node"] = node_ix
    conf["port"] = port

    print(f"Serving on {port}")
    ts = TrainerServer(conf)
    ts.fit()

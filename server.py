import train_pb2_grpc
import train_pb2
from concurrent import futures
import grpc

class TrainerServer(train_pb2_grpc.TrainerServicer):
    def __init__(self):
        pass

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
        resp = train_pb2.Grads(epoch=request.epoch, step=request.step, data=output_data)
        return resp

def serve(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    train_pb2_grpc.add_TrainerServicer_to_server(
        TrainerServer(), server
    )
    addr = f"[::]:{port}"
    server.add_insecure_port(addr)
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    port = 50051
    print("Serving on 50051")
    serve(port)
import train_pb2
import train_pb2_grpc
import grpc


def run_train_step(stub):
    input = "Is it working?"
    data = str.encode(input)
    req = train_pb2.Weights(epoch=0, step=1, data=data)
    resp = stub.RunStep(req)
    output = resp.data.decode()
    print(output)
    return

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = train_pb2_grpc.TrainerStub(channel)
        print("-------------- RunTrainStep --------------")
        run_train_step(stub)
        


if __name__ == "__main__":
    run()
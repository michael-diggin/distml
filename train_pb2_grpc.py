# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import train_pb2 as train__pb2


class TrainerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RunStep = channel.unary_unary(
                '/train.Trainer/RunStep',
                request_serializer=train__pb2.RunStepReuest.SerializeToString,
                response_deserializer=train__pb2.RunStepResponse.FromString,
                )
        self.Finish = channel.unary_unary(
                '/train.Trainer/Finish',
                request_serializer=train__pb2.FinishRequest.SerializeToString,
                response_deserializer=train__pb2.FinishResponse.FromString,
                )


class TrainerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def RunStep(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Finish(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TrainerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RunStep': grpc.unary_unary_rpc_method_handler(
                    servicer.RunStep,
                    request_deserializer=train__pb2.RunStepReuest.FromString,
                    response_serializer=train__pb2.RunStepResponse.SerializeToString,
            ),
            'Finish': grpc.unary_unary_rpc_method_handler(
                    servicer.Finish,
                    request_deserializer=train__pb2.FinishRequest.FromString,
                    response_serializer=train__pb2.FinishResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'train.Trainer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Trainer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def RunStep(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/train.Trainer/RunStep',
            train__pb2.RunStepReuest.SerializeToString,
            train__pb2.RunStepResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Finish(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/train.Trainer/Finish',
            train__pb2.FinishRequest.SerializeToString,
            train__pb2.FinishResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

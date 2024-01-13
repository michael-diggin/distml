# file containing helper functions for converting
# TF tensors (grads) and numpy ndarrays (weights) to proto and back

import train_pb2
import numpy as np
import tensorflow as tf

def weights_to_proto(weights):
    return [train_pb2.Ndarray(data=w.tobytes(), dtype=np_type_to_proto(w.dtype), shape=list(w.shape)) for w in weights]

def weights_from_proto(w_proto):
    return [np.frombuffer(w.data, dtype=np_type_from_proto(w.dtype)).reshape(w.shape) for w in w_proto]

def grads_to_proto(grads):
    return [train_pb2.TFTensor(data=tf.io.serialize_tensor(g).numpy(), dtype=tf_type_to_proto(g.dtype)) for g in grads]

def grads_from_proto(g_proto):
    return [tf.io.parse_tensor(g.data, out_type=tf_type_from_proto(g.dtype)) for g in g_proto]

def np_type_to_proto(dtype):
    if dtype == "float32":
        return train_pb2.DataType.FLOAT_32
    if dtype == "float64":
        return train_pb2.DataType.FLOAT_64
    print(f"Unknown np dtype: {dtype}")
    return train_pb2.DataType.UNKNOWN

def tf_type_to_proto(dtype):
    if dtype == "float32":
        return train_pb2.DataType.FLOAT_32
    if dtype == "float64":
        return train_pb2.DataType.FLOAT_64
    print(f"Unknown tf dtype: {dtype}")
    return train_pb2.DataType.UNKNOWN

def np_type_from_proto(dtype):
    if dtype == train_pb2.DataType.FLOAT_32:
        return np.float32
    if dtype == train_pb2.DataType.FLOAT_64:
        return np.float64
    # should somehow handle this weird case
    print(f"Unhandled dtype: {dtype}")
    return np.float32

def tf_type_from_proto(dtype):
    if dtype == train_pb2.DataType.FLOAT_32:
        return tf.float32
    if dtype == train_pb2.DataType.FLOAT_64:
        return tf.float64
    print(f"Unhandled tf dtype: {dtype}")
    return tf.float32
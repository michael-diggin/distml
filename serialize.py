# file containing helper functions for converting
# TF tensors (grads) and numpy ndarrays (weights) to proto and back
import logging

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

def loss_to_proto(loss):
    return train_pb2.TFTensor(data=tf.io.serialize_tensor(loss).numpy(), dtype=tf_type_to_proto(loss.dtype))

def loss_from_proto(l_proto):
    return tf.io.parse_tensor(l_proto.data, out_type=tf_type_from_proto(l_proto.dtype))

def opt_weights_to_proto(opt_weights):
    pw = []
    for ow in opt_weights:
        if type(ow) != np.ndarray:
            ow = np.array(ow)
        pw.append(train_pb2.Ndarray(data=ow.tobytes(), dtype=np_type_to_proto(ow.dtype), shape=list(ow.shape)))
    return pw

def opt_weights_from_proto(opt_proto):
    ow = []
    for pw in opt_proto:
        data = np.frombuffer(pw.data, dtype=np_type_from_proto(pw.dtype))
        if pw.shape == []:
            ow.append(data.item())
        else:
            ow.append(data.reshape(pw.shape))
    return ow


def np_type_to_proto(dtype):
    if dtype == "float32":
        return train_pb2.DataType.FLOAT_32
    if dtype == "float64":
        return train_pb2.DataType.FLOAT_64
    if dtype == "int32":
        return train_pb2.DataType.INT_32
    if dtype == "int64":
        return train_pb2.DataType.INT_64
    logging.warning(f"Unknown np dtype: {dtype}")
    return train_pb2.DataType.UNKNOWN

def tf_type_to_proto(dtype):
    if dtype == "float32":
        return train_pb2.DataType.FLOAT_32
    if dtype == "float64":
        return train_pb2.DataType.FLOAT_64
    logging.warning(f"Unknown tf dtype: {dtype}")
    return train_pb2.DataType.UNKNOWN

def np_type_from_proto(dtype):
    if dtype == train_pb2.DataType.FLOAT_32:
        return np.float32
    if dtype == train_pb2.DataType.FLOAT_64:
        return np.float64
    if dtype == train_pb2.DataType.INT_32:
        return np.int32
    if dtype == train_pb2.DataType.INT_64:
        return np.int64
    logging.warning(f"Unhandled dtype: {dtype}")
    return np.float32

def tf_type_from_proto(dtype):
    if dtype == train_pb2.DataType.FLOAT_32:
        return tf.float32
    if dtype == train_pb2.DataType.FLOAT_64:
        return tf.float64
    logging.warning(f"Unhandled tf dtype: {dtype}")
    return tf.float32
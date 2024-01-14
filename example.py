from distml import TrainerServer
import os
import numpy as np
import tensorflow as tf


def create_dataset(num):
    # w = 3, b = 0.5
    X = np.random.rand(num, 1)
    Y = 3*X + 0.5 + np.random.randn(num, 1)*0.01
    return X, Y

def create_example_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    return model



if __name__ == "__main__":
    conf = {
        "leader": "0",
        "servers": ["localhost:1234"]#, "localhost:1235"],
    }
    node_ix = os.environ.get("NODE", "")
    port = os.environ.get("PORT", 1234)
    conf["node"] = node_ix
    conf["port"] = port

    tf.keras.utils.set_random_seed(111)
    X, Y = create_dataset(10)
    model = create_example_model()
    loss_func = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    print(f"Serving on {port}")
    ts = TrainerServer(conf, model, loss_func, opt)
    ts.fit(10, None, X, Y)
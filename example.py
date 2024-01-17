from distml import TrainerServer
import os
import numpy as np
import tensorflow as tf
import checkpoint


def create_dataset(num):
    # w = 3, b = 0.5
    X = np.random.rand(num, 1)
    Y = 3*X + 0.5 + np.random.randn(num, 1)*0.005
    return X, Y

def create_example_model():
    model = tf.keras.Sequential([
         tf.keras.layers.InputLayer(shape=(1,)),
         tf.keras.layers.Dense(1)])
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
    X, Y = create_dataset(1000)
    model = create_example_model()
    
    loss_func = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)

    checkpointer = checkpoint.FileCheckpoint("chpt", 10)

    print(f"Serving on {port}")
    ts = TrainerServer(conf, model, loss_func, opt, checkpointer)
    ts.fit(100, 64, X, Y)

    print(ts.model.trainable_variables[0].numpy()) # should be near 3
    print(ts.model.trainable_variables[1].numpy()) # should be near 0.5

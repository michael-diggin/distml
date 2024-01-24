from distml import TrainerServer
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import checkpoint

def create_mnist_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Input((784,)),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def get_dataset(size, test_size=1000):
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    shuffled_idx = [i for i in range(train_labels.shape[0])]
    np.random.shuffle(shuffled_idx)

    train_images = train_images[shuffled_idx]
    train_labels = train_labels[shuffled_idx]

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    # reshape
    train_images = np.reshape(train_images, (-1, 784))
    test_images = np.reshape(test_images, (-1, 784))

    # normalize
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    return train_images[0:size, :], train_labels[0:size, :], test_images[0:test_size, :], test_labels[0:test_size, :]

def test_accuracy(preds, true_labels):
    maxpos = lambda x : np.argmax(x)
    y_test_max = np.array([maxpos(rec) for rec in true_labels])
    pred_max = np.array([maxpos(rec) for rec in preds])

    cal_acc = sum(pred_max == y_test_max)/len(pred_max)
    return cal_acc


if __name__ == '__main__':
    conf = {
        "leader": "0",
        "servers": ["localhost:1231", "localhost:1232"],
    }
    node_ix = os.environ.get("NODE", "")
    port = os.environ.get("PORT", 1234)
    conf["node"] = node_ix
    conf["port"] = port

    tf.keras.utils.set_random_seed(123)

    x_train, y_train, x_test, y_test = get_dataset(size=30000, test_size=1000)
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shard(3, int(node_ix))

    model = create_mnist_model()
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    checkpointer = checkpoint.FileCheckpoint("chpt", 10)

    print(f"Running on {port}")
    ts = TrainerServer(conf, model, loss_func, opt, checkpointer)
    # step size = 32*3
    # num_steps = floor(30,000/96)
    ts.fit(epochs=15, dataset=ds, num_steps=312)

    pred_test = model(x_test)
    cal_acc = test_accuracy(pred_test, y_test)
    print(f"Test set accuracy: {cal_acc}")
    # Achieves ~95.8% accuracy on the test set
    



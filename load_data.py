from keras.datasets import mnist
import numpy as np

np.random.seed(3)

# Data preparation
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return format_data(x_train, y_train, 10), format_data(x_test, y_test, 10)


def format_data(x, y, n_hot=None):
    if n_hot:
        y = onehot(y, n_hot)
    x_r = flatten(reshape_m(x))
    return x_r, y


def load_binary_class_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    idx_train = y_train < 2
    idx_test = y_test < 2
    return format_data(x_train[idx_train], y_train[idx_train]), format_data(x_test[idx_test], y_test[idx_test])


def onehot(y, n_c):
    m = y.shape[0]
    y_hot = np.zeros((n_c, m))
    y_hot[y, np.arange(m)] = 1
    return y_hot


def reshape_m(x):
    return np.transpose(x, (1, 2, 0))


def flatten(x):
    a, b, m = x.shape
    return np.reshape(x, (a * b, m))


import numpy as np

np.random.seed(3)


import os
data_path = 'mnist/'
file_path = data_path + 'mnist_data.npy'
if not os.path.exists(data_path):
    os.makedirs(data_path)

if os.path.isfile(file_path):
    mnist_data = np.load(file_path)
else:
    from keras.datasets import mnist
    mnist_data = mnist.load_data()
    np.save(file_path, mnist_data)

(x_train, y_train), (x_test, y_test) = mnist_data


# Data preparation
def load_data():
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return load_class_data(10)


def load_class_data(num_classes=None, should_onehot=True, should_flatten=True):
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if num_classes is None:
        num_classes = np.max(y_train)
    idx_train = y_train < num_classes
    idx_test = y_test < num_classes
    return format_data(x_train[idx_train], y_train[idx_train], num_classes, should_onehot, should_flatten), format_data(x_test[idx_test], y_test[idx_test], num_classes, should_onehot, should_flatten)


def format_data(x, y, n_classes, should_onehot=True, should_flatten=True):
    if should_onehot:
        y = onehot(y, n_classes)
    x_r = reshape_m(x)
    if should_flatten:
        x_r = flatten(x_r)
    return x_r, y


def load_binary_class_data():
    return load_class_data(2, should_onehot=False)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # idx_train = y_train < 2
    # idx_test = y_test < 2
    # return format_data(x_train[idx_train], y_train[idx_train], 2), format_data(x_test[idx_test], y_test[idx_test])


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


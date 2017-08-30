import numpy as np
from keras.datasets import mnist


# Data preparation
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train_hot = onehot(y_train, 10)
    y_test_hot = onehot(y_test, 10)
    x_train_r = flatten(reshape_m(x_train))
    x_test_r = flatten(reshape_m(x_test))
    return (x_train_r, y_train_hot), (x_test_r, y_test_hot)


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


# Model setup

def initialize_weights(n_x, n_h):
    w = np.random.randn(n_h, n_x) * xavier_initialization(n_x)
    b = np.zeros((n_h, 1), dtype=np.float32)
    return (w, b)


def xavier_initialization(n_x):
    return 2 / n_x


def linear(w, x, b):
    # (n_h, n_x) * (n_x, m) + (n_h, 1) = (n_h, m)
    return np.dot(w, x) + b


def linear_d(dz, w, a_prev, b):
    _, m = a_prev.shape

    # b = (n_h, 1)
    # w = (n_h, n_x)
    # dz = (n_h, m)
    # a = (n_x, m)

    # WARNING: make sure to check these dimensions
    da_prev = np.dot(w.T, dz)   # (n_x, n_h) * (n_h, m)
    dw = 1 / m * np.dot(dz, a_prev.T)    # (n_h, m) * (m, n_x)
    db = np.mean(dz, axis=1, keepdims=True)     # (n_h, m) / m
    return da_prev, dw, db


def relu(z):
    return np.maximum(z, 0)


def relu_d(a):
    return a > 0


def softmax(z):
    # Shift z values so highest value is 0
    # Must stabilize as exp can get out of control
    z_norm = z - np.max(z)
    exp = np.exp(z_norm)
    return exp / np.sum(exp, axis=0, keepdims=True)


def softmax_d(z):
    return z * (1 - z)


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoid_d():
    return None


def forward_pass(w, b, X, Y):
    return None


def backpropagate(x, b, X, Y):
    return None


def compute_cost(Y, z):
    return - np.mean(Y * np.log(z) + (1 - Y) * np.log(1 - z))


def categorical_cross_entropy(Y, a3):
    return - np.mean(Y * np.log(a3) + (1 - Y) * np.log(1 - a3))


def categorical_cross_entropy_d(Y, a3):
    cost_d = Y / a3 + (1 - Y) / (1 - a3)
    return - cost_d

# Let's create a model with 2 hidden layers with 100 units

def model(X_train, Y_train, X_test, Y_test, num_interations=2000, learning_rate=0.5):
    n_x, n_m = X_train.shape
    n_y, _ = Y_train.shape
    n_h1, n_h2 = [100, 100]

    w1, b1 = initialize_weights(n_x, n_h1)
    w2, b2 = initialize_weights(n_h1, n_h2)
    w3, b3 = initialize_weights(n_y, n_h2)

    # forward pass
    z1 = linear(w1, X_train, b1)
    a1 = relu(z1)

    z2 = linear(w2, a1, b2)
    a2 = relu(z2)

    z3 = linear(w3, a2, b3)
    a3 = softmax(z3)

    # Cost
    cost = categorical_cross_entropy(Y_train, a3)
    print('Cost:', cost)

    da3_step = categorical_cross_entropy_d(Y_train, a3)
    dz3_step = softmax_d(z3) * da3_step

    dz3 = a3 - Y_train

    assert(np.all(dz3_step == dz3))

    da2, dw3, db3 = linear_d(dz3, w3, a2, b3)
    dz2 = relu_d(a2) * da2

    da1, dw2, db2 = linear_d(dz2, w2, a1, b2)
    dz1 = relu_d(a1)


    _, dw1, db1 = linear_d(dz1, w1, X_train, b1)

    return None


(x_train, y_train), (x_test, y_test) = load_data()

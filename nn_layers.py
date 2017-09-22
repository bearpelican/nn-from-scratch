import numpy as np
import load_data
import copy

class 

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
    # b = (n_h, 1)
    # w = (n_h, n_x)
    # dz = (n_h, m)
    # a = (n_x, m)
    _, m = a_prev.shape

    da_prev = np.dot(w.T, dz)   # (n_x, n_h) * (n_h, m)
    dw = 1 / m * np.dot(dz, a_prev.T)    # (n_h, m) * (m, n_x)
    db = np.mean(dz, axis=1, keepdims=True)     # (n_h, m) / m
    return da_prev, dw, db


def relu(z):
    return np.maximum(z, 0)


def relu_d(a):
    return np.int64(a > 0)


def softmax(z):
    # Shift z values so highest value is 0
    # Must stabilize as exp can get out of control
    z_norm = z - np.max(z)
    exp = np.exp(z_norm)
    return exp / np.sum(exp, axis=0, keepdims=True)


def softmax_d(z):
    # No idea how to implement this. See softmax.py
    return None

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoid_d(z):
    # a must be sigmoid activated
    return z * (1 - z)

#
def compute_cost(Y, z):
    return - np.mean(Y * np.log(z) + (1 - Y) * np.log(1 - z))


def categorical_cross_entropy(y, a):
    cost = np.sum(y * np.log(a), axis=1, keepdims=True)
    return - np.mean(cost)

def categorical_cross_entropy_d(y, a3):
    # cost_d = y / a3 + (1 - y) / (1 - a3)
    # return - cost_d
    return - (y / a3)


def binary_cross_entropy(y, a):
    cost = y * np.log(a) + (1 - y) * np.log(1 - a)
    return - np.mean(cost)


def binary_cross_entropy_d(y, a):
    # cost_d = y / a + (1 - y) / (1 - a)
    cost_d = y - a / (y * (1 - y))  # same as above
    return - cost_d

def forward_pass(X, Y, weights):
    w1, b1, w2, b2, w3, b3 = weights
    # forward pass
    z1 = linear(w1, X, b1)
    a1 = relu(z1)

    z2 = linear(w2, a1, b2)
    a2 = relu(z2)

    z3 = linear(w3, a2, b3)
    a3 = softmax(z3)

    # Cost
    cost = categorical_cross_entropy(Y, a3)
    return (cost, (z1, a1, z2, a2, z3, a3))



def backpropagate(X, Y, weights, activations):
    w1, b1, w2, b2, w3, b3 = weights
    z1, a1, z2, a2, z3, a3 = activations

    dz3 = a3 - Y

    da2, dw3, db3 = linear_d(dz3, w3, a2, b3)
    dz2 = relu_d(a2) * da2

    da1, dw2, db2 = linear_d(dz2, w2, a1, b2)
    dz1 = relu_d(a1) * da1

    _, dw1, db1 = linear_d(dz1, w1, X, b1)
    return dw1, db1, dw2, db2, dw3, db3

# Let's create a model with 2 hidden layers with 100 units
def model(X_train, Y_train, X_test, Y_test, num_iterations=50, learning_rate=0.01):
    n_x, n_m = X_train.shape
    n_y, _ = Y_train.shape
    # n_y = 1
    n_h1, n_h2 = [100, 100]

    w1, b1 = initialize_weights(n_x, n_h1)
    w2, b2 = initialize_weights(n_h1, n_h2)
    w3, b3 = initialize_weights(n_h2, n_y)

    for i in range(num_iterations):
        # forward pass
        weights = w1, b1, w2, b2, w3, b3
        cost, activations = forward_pass(X_train, Y_train, weights)
        print('Cost:', cost)

        gradients = backpropagate(X_train, Y_train, weights, activations)
        dw1, db1, dw2, db2, dw3, db3 = gradients

        assert(dw3.shape == w3.shape)
        assert(dw2.shape == w2.shape)
        assert(dw1.shape == w1.shape)

        # Update weights
        w3 -= learning_rate * dw3
        b3 -= learning_rate * db3
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

    # Accuracy
    weights = w1, b1, w2, b2, w3, b3
    cost, activations = forward_pass(X_test, Y_test, weights)
    z1, a1, z2, a2, z3, a3 = activations
    # pred = np.round(a3)

    # this is for cross entropy
    pred = np.zeros(a3.shape)
    pred[a3.argmax(axis=0), np.arange(a3.shape[1])] = 1

    acc = np.mean(pred == Y_test)
    # print(pred == Y_test)
    print('Accuracy:', acc)

    return acc



def gradient_check(X, Y):
    n_x, n_m = X.shape
    # n_y, _ = Y_train.shape
    n_y = 1
    n_h1, n_h2 = [10, 10]

    w1, b1 = initialize_weights(n_x, n_h1)
    w2, b2 = initialize_weights(n_h1, n_h2)
    w3, b3 = initialize_weights(n_h2, n_y)

    weights = w1, b1, w2, b2, w3, b3
    cost1, activations = forward_pass(X, Y, weights)
    gradients = backpropagate(X, Y, weights, activations)
    approx_gradients = copy.deepcopy(gradients)

    # Gradient checking
    epsilon = .00001
    all_weights = (w1, b1, w2, b2, w3, b3)
    num_parameters = len(all_weights)

    for i in range(num_parameters):
        current_param = all_weights[i]

        for row in range(current_param.shape[0]):
            for col in range(current_param.shape[1]):
                thetaplus = copy.deepcopy(all_weights)
                thetaminus = copy.deepcopy(all_weights)

                thetaplus[i][row, col] = (thetaplus[i][row, col] + epsilon)
                thetaminus[i][row, col] = (thetaminus[i][row, col] - epsilon)

                J_plus, _ = forward_pass(X, Y, thetaplus)
                J_minus, _ = forward_pass(X, Y, thetaminus)

                approx = (J_plus - J_minus) / (2 * epsilon)
                approx_gradients[i][row, col] = approx
        print('Completed param:', i)

    def euclidean(x):
        return np.sqrt(np.sum(x ** 2))

    def flat_array(x):
        res = np.array([])
        for i in range(len(x)):
            res = np.concatenate((res, x[i].flatten()))
        return res

    np_gradients = flat_array(gradients)
    np_gradients_approx = flat_array(approx_gradients)
    numerator = euclidean(np.array(np_gradients) - np.array(np_gradients_approx))
    denominator = euclidean(np_gradients) + euclidean(np_gradients_approx)
    difference = numerator / denominator
    return difference

# (x_train, y_train), (x_test, y_test) = load_data.load_binary_class_data()
# model(x_train[:, :100], y_train[:100], x_test[:, :100], y_test[:100])

# gradient_check(x_train[:, :100], y_train[:100])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[:, 1].reshape(28, 28))

(x_train, y_train), (x_test, y_test) = load_data.load_class_data(10)
model(x_train, y_train, x_test, y_test)
# model(x_train[:, :1000], y_train[:, :1000], x_test[:, :1000], y_test[:, :1000])


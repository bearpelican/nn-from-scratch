import numpy as np
import load_data

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
    # exps = np.exp(z)
    # others = exps.sum() - exps
    # return 1 / (2 + exps / others + others / exps)

    #
    # for i in range(len(self.value)):
    #     for j in range(len(z)):
    #         if i == j:
    #             self.gradient[i] = self.value[i] * (1-z[i))
    #         else:
    #              self.gradient[i] = -self.value[i]*z[j]
    SM = z.reshape((-1,1))
    jac = np.diag(z) - np.dot(SM, SM.T)
    return jac

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def sigmoid_d(z):
    # a must be sigmoid activated
    return z * (1 - z)


def compute_cost(Y, z):
    return - np.mean(Y * np.log(z) + (1 - Y) * np.log(1 - z))


def categorical_cross_entropy(y, a):
    cost = np.sum(y * np.log(a), axis=1, keepdims=True)
    print('Categorical cross entropy:', cost.shape)
    return - np.mean(cost)

def categorical_cross_entropy_d(y, a3):
    # cost_d = y / a3 + (1 - y) / (1 - a3)
    # return - cost_d
    return - (y / a3)


def binary_cross_entropy(y, a):
    cost = y * np.log(a) + (1 - y) * np.log(1 - a)
    print('Categorical cross entropy:', cost.shape)
    return - np.mean(cost)


def binary_cross_entropy_d(y, a, z):
    # d = (a - y) * z
    # return np.mean(d)
    cost_d = y / a + (1 - y) / (1 - a)
    # cost_d = y - a / (y * (1 - y))
    return - cost_d

# def forward_pass(X, weights, layer_dims):
#     for i in range(len(layer_dims)):
#         return None



def backpropagate(x, b, X, Y):
    return None

# Let's create a model with 2 hidden layers with 100 units

def model(X_train, Y_train, X_test, Y_test, num_iterations=5, learning_rate=0.5):
    n_x, n_m = X_train.shape
    # n_y, _ = Y_train.shape
    n_y = 1
    n_h1, n_h2 = [100, 100]

    w1, b1 = initialize_weights(n_x, n_h1)
    w2, b2 = initialize_weights(n_h1, n_h2)
    w3, b3 = initialize_weights(n_h2, n_y)

    for i in range(num_iterations):
        # forward pass
        z1 = linear(w1, X_train, b1)
        a1 = relu(z1)

        z2 = linear(w2, a1, b2)
        a2 = relu(z2)

        z3 = linear(w3, a2, b3)
        a3 = sigmoid(z3)

        # Cost
        cost = binary_cross_entropy(Y_train, a3)
        print('Cost:', cost)

        dcost_step = binary_cross_entropy_d(Y_train, a3, z3)
        dz3_step = sigmoid_d(a3) * dcost_step


        dz3 = a3 - Y_train

        # Gradient checking
        epsilon = .000001
        w3_gcp = w3 + epsilon
        b3_gcp = b3 + epsilon
        z3_gcp = linear(w3_gcp, a2, b3_gcp)
        a3_gcp = sigmoid(z3_gcp)
        cost_gcp = binary_cross_entropy(Y_train, a3_gcp)

        w3_gcm = w3 - epsilon
        b3_gcm = b3 - epsilon
        z3_gcm = linear(w3_gcm, a2, b3_gcm)
        a3_gcm = sigmoid(z3_gcm)
        cost_gcm = binary_cross_entropy(Y_train, a3_gcm)

        dz3_gc = (z3_gcp - z3_gcm) / (2 * epsilon)
        da3_gc = (a3_gcp - a3_gcm) / (2 * epsilon)
        dcost_gc = (cost_gcp - cost_gcm) / (2 * epsilon)
        distance = np.sum((dz3 - dz3_gc) ** 2)
        print('Distance:', distance)
        print('Difference:', dz3 - dz3_gc)




        print('Dz3 Step:', dz3_step.shape)
        print('Dz3:', dz3.shape)
        # assert(np.all(dz3_step == dz3))

        da2, dw3, db3 = linear_d(dz3, w3, a2, b3)
        dz2 = relu_d(a2) * da2

        da1, dw2, db2 = linear_d(dz2, w2, a1, b2)
        dz1 = relu_d(a1)

        _, dw1, db1 = linear_d(dz1, w1, X_train, b1)

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
    z1 = linear(w1, X_test, b1)
    a1 = relu(z1)

    z2 = linear(w2, a1, b2)
    a2 = relu(z2)

    z3 = linear(w3, a2, b3)
    a3 = softmax(z3)

    pred = np.zeros(a3.shape)
    pred[a3.argmax(axis=0), np.arange(a3.shape[1])] = 1

    acc = np.mean(pred == Y_test)
    print(pred.shape)
    print(Y_test.shape)
    # print(pred == Y_test)
    print('Accuracy:', acc)

    return None


(x_train, y_train), (x_test, y_test) = load_data.load_binary_class_data()

# import matplotlib.pyplot as plt
# plt.imshow(x_train[:, 1].reshape(28, 28))
model(x_train[:, :100], y_train[:100], x_test[:, :100], y_test[:100])


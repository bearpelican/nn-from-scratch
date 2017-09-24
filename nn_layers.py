import numpy as np
import load_data
import copy


class NNLayer:
    # n_x = number of inputs (from previous layer)
    # n_h = number of hidden units per layer
    def __init__(self, n_x, n_h):
        self.n_x = n_x
        self.n_h = n_h
        self.w, self.b = initialize_weights(n_x, n_h)
        self.z = None
        self.a = None
        self.a_prev = None
        self.dw = None
        self.db = None
        self.dz = None

    def forward(self, a_prev):
        # z = linear computation
        # a = activation
        self.z = linear(self.w, a_prev, self.b)
        self.a = self.activation(self.z)
        self.a_prev = a_prev
        return self.a

    def activation(self, z):
        raise NotImplementedError  # you want to override this on the child classes

    def linear_d(self, l_next):
        raise NotImplementedError  # you want to override this on the child classes

    def update_weights(self):
        pass

    def backward(self, l_next):
        # use activation from previous layer and return new activation
        # save a and z values
        dz = self.linear_d(l_next)
        _, m = self.a_prev.shape
        self.dw = 1 / m * np.dot(dz, self.a_prev.T)  # (n_h, m) * (m, n_x)
        self.db = np.mean(dz, axis=1, keepdims=True)  # (n_h, m) / m
        self.dz = dz

        # dz3 = a3 - Y
        # _, m = a2.shape
        # dw3 = 1 / m * np.dot(dz3, a2.T)  # (n_h, m) * (m, n_x)
        # db3 = np.mean(dz3, axis=1, keepdims=True)  # (n_h, m) / m
        # # da2, dw3, db3 = linear_d(dz3, w3, a2)
        #
        # _, m = a1.shape
        # da2 = np.dot(w3.T, dz3)  # (n_x, n_h) * (n_h, m)
        # dz2 = relu_d(a2) * da2
        # dw2 = 1 / m * np.dot(dz2, a1.T)  # (n_h, m) * (m, n_x)
        # db2 = np.mean(dz2, axis=1, keepdims=True)  # (n_h, m) / m
        # # da1, dw2, db2 = linear_d(dz2, w2, a1)
        #
        # _, m = X.shape
        # da1 = np.dot(w2.T, dz2)  # (n_x, n_h) * (n_h, m)
        # dz1 = relu_d(a1) * da1
        # dw1 = 1 / m * np.dot(dz1, X.T)  # (n_h, m) * (m, n_x)
        # db1 = np.mean(dz1, axis=1, keepdims=True)  # (n_h, m) / m
        # # _, dw1, db1 = linear_d(dz1, w1, X)


class Unit:
    def __init__(self, n_x, n_h):
        self.n_x = n_x
        self.n_h = n_h

    def activation(self, z):
        raise NotImplementedError  # you want to override this on the child classes

    def derivative(self, z):
        raise NotImplementedError  # you want to override this on the child classes


class LinearUnit(Unit):
    def __init__(self, n_x, n_h):
        Unit.__init__(self, n_x, n_h)
        self.w, self.b = initialize_weights(n_x, n_h)
        self.z = None
        self.x = None # a_prev
        self.dw = None
        self.db = None
        # self.dz = None # dz is calculated by activation layer
        self.dx = None # a_prev

    def activation(self, x):
        # (n_h, n_x) * (n_x, m) + (n_h, 1) = (n_h, m)
        def linear(w, x, b):
            # (n_h, n_x) * (n_x, m) + (n_h, 1) = (n_h, m)
            return np.dot(w, x) + b
        self.x = x
        self.z = linear(self.w, x, self.b)
        return self.z

    def derivative(self, dz):
        def linear_d(_dz, w, x):
            # b = (n_h, 1) - bias is always dz. no need to pass in as param
            # w = (n_h, n_x)
            # dz = (n_h, m)
            # x = (n_x, m) AKA a_previous
            _, m = x.shape

            dx = np.dot(w.T, _dz)  # (n_x, n_h) * (n_h, m)
            dw = 1 / m * np.dot(_dz, x.T)  # (n_h, m) * (m, n_x)
            db = np.mean(_dz, axis=1, keepdims=True)  # (n_h, m) / m
            return dx, dw, db
        self.dx, self.dw, self.db = linear_d(dz, self.w, self.dx)
        return self.dx


class ActivationUnit(Unit):
    def __init__(self, n_x, n_h):
        Unit.__init__(self, n_x, n_h)
        self.a = None
        self.da = None

        # dz is in the Activation Unit instead of Linear Unit
        # This makes it easier to calculate since it depends on a and da
        # Also, we the type of multiplication depends on the activation - See differences between RELU and Softmax
        self.dz = None

    def activation(self, z):
        raise NotImplementedError  # you want to override this on the child classes

    def derivative(self, da): # AKA dx of linear layer
        raise NotImplementedError  # you want to override this on the child classes


class RELU(ActivationUnit):
    def activation(self, z):
        def relu(_z):
            return np.maximum(_z, 0)
        self.a = relu(z)
        return self.a

    def derivative(self, da):
        def relu_d(a):
            return np.int64(a > 0)
        self.dz = da * relu_d(self.a)
        return self.dz


class Sigmoid(ActivationUnit):
    def activation(self, z):
        def sigmoid(_z):
            return 1 / (1 + np.exp(-_z))
        self.a = sigmoid(z)
        return self.a

    def derivative(self, da):
        def sigmoid_d(a):
            return a * (1 - a)
        self.dz = da * sigmoid_d(self.a)
        return self.dz


class Softmax(ActivationUnit):
    def activation(self, z):
        def softmax(_z):
            # Shift z values so highest value is 0
            # Must stabilize as exp can get out of control
            z_norm = _z - np.max(_z)
            exp = np.exp(z_norm)
            return exp / np.sum(exp, axis=0, keepdims=True)

        self.a = softmax(z)
        return self.a

    def derivative(self, da):
        # (n_class, n_class, n_m_examples)
        # Finds softmax for m training examples
        def softmax_d(a):
            # Softmax derivative function (Jacobian)
            def softmax_grad(softmax):
                s = softmax.reshape(-1, 1)
                return np.diagflat(s) - np.dot(s, s.T)
            # Find softmax for each m example
            n_class, n_m = a.shape
            s_grad = np.empty((n_class, n_class, n_m))
            for i in range(a.shape[1]):
                s_grad[:, :, i] = softmax_grad(a[:, i])
            return s_grad
        s_d = softmax_d(self.a)
        self.dz = np.einsum('ijk,jk->ik', s_d, da)
        return self.dz


class Cost:
    def cost(self, y, a):
        # y = target (actual truth)
        # a = prediction
        raise NotImplementedError  # you want to override this on the child classes

    def cost_d(self, y, a):
        # y = target (actual truth)
        # a = prediction
        raise NotImplementedError  # you want to override this on the child classes


class CategoricalCrossEntropy(Cost):
    def cost(self, y, a):
        def categorical_cross_entropy(_y, _a):
            cost = np.sum(_y * np.log(_a), axis=1, keepdims=True)
            return - np.mean(cost)
        return categorical_cross_entropy(y, a)

    def cost_d(self, y, a):
        def categorical_cross_entropy_d(_y, _a):
            return - (_y / _a)
        return categorical_cross_entropy_d(y, a)


class BinaryCrossEntropy(Cost):
    def cost(self, y, a):
        def binary_cross_entropy(_y, _a):
            cost = _y * np.log(_a) + (1 - _y) * np.log(1 - _a)
            return - np.mean(cost)
        return binary_cross_entropy(y, a)

    def cost_d(self, y, a):
        def binary_cross_entropy_d(_y, _a):
            # cost_d = y / a + (1 - y) / (1 - a)
            cost_d = _y - _a / (_y * (1 - _y))  # same as above
            return - cost_d

        return binary_cross_entropy_d(y, a)


class HiddenLayer(NNLayer):
    def linear_d(self, l_next):
        w_next = l_next.w
        dz_next = l_next.dz
        da = np.dot(w_next.T, dz_next)  # (n_x, n_h) * (n_h, m)
        dz = self.activation_d(self.a) * da
        return dz

    def activation(self, z):
        raise NotImplementedError  # you want to override this on the child classes

    def activation_d(self, a):
        raise NotImplementedError  # you want to override this on the child classes


class OutputLayer(NNLayer):

    def cost(self, y):
        self.y = y
        raise NotImplementedError  # you want to override this on the child classes

    def linear_d(self, l_next):
        return Y - self.a



def initialize_weights(n_x, n_h):
    w = np.random.randn(n_h, n_x) * xavier_initialization(n_x)
    b = np.zeros((n_h, 1), dtype=np.float32)
    return w, b


def xavier_initialization(n_x):
    return 2 / n_x


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


def backprop(X, Y, weights, activations):
    w1, b1, w2, b2, w3, b3 = weights
    z1, a1, z2, a2, z3, a3 = activations

    dz3 = a3 - Y

    cost_d = categorical_cross_entropy_d(Y, a3)
    a3_d = softmax_d_m(a3)
    print('A3', a3.shape)
    print(cost_d.shape)
    print(a3_d.shape)
    cost_d_r = cost_d.reshape((cost_d.shape[0], 1, cost_d.shape[1]))
    dz3_step = np.einsum('ijk,jyk->iyk', a3_d, cost_d_r)
    dz3_step_r = dz3_step.reshape((dz3_step.shape[0], dz3_step.shape[2]))


    _, m = a2.shape
    dw3 = 1 / m * np.dot(dz3, a2.T)    # (n_h, m) * (m, n_x)
    db3 = np.mean(dz3, axis=1, keepdims=True)     # (n_h, m) / m
    da2 = np.dot(w3.T, dz3)   # (n_x, n_h) * (n_h, m)
    # da2, dw3, db3 = linear_d(dz3, w3, a2)

    _, m = a1.shape
    dz2 = relu_d(a2) * da2
    dw2 = 1 / m * np.dot(dz2, a1.T)    # (n_h, m) * (m, n_x)
    db2 = np.mean(dz2, axis=1, keepdims=True)     # (n_h, m) / m
    da1 = np.dot(w2.T, dz2)   # (n_x, n_h) * (n_h, m)
    # da1, dw2, db2 = linear_d(dz2, w2, a1)

    _, m = X.shape
    dz1 = relu_d(a1) * da1
    dw1 = 1 / m * np.dot(dz1, X.T)    # (n_h, m) * (m, n_x)
    db1 = np.mean(dz1, axis=1, keepdims=True)     # (n_h, m) / m
    # _, dw1, db1 = linear_d(dz1, w1, X)

    return dw1, db1, dw2, db2, dw3, db3

def backpropagate(X, Y, weights, activations):
    w1, b1, w2, b2, w3, b3 = weights
    z1, a1, z2, a2, z3, a3 = activations

    dz3 = a3 - Y
    da2, dw3, db3 = linear_d(dz3, w3, a2)

    dz2 = relu_d(a2) * da2
    da1, dw2, db2 = linear_d(dz2, w2, a1)

    dz1 = relu_d(a1) * da1
    _, dw1, db1 = linear_d(dz1, w1, X)

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


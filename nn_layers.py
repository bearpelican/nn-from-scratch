import numpy as np
import load_data
import copy


class NNLayer:
    """NN forward propagation

    Attributes:
        n_x: number of inputs (from previous layer)
        n_h: number of hidden units per layer
        activation: activation function
    """
    def __init__(self, n_x, n_h, activation):
        self.n_x = n_x
        self.n_h = n_h
        self.linear_unit = LinearUnit(n_x, n_h)
        self.activation = activation
        # self.learning_rate = learning_rate

    def forward(self, x):
        """NN forward propagation

        Attributes:
            x (n_x, n_h, n_m) matrix: layer input
        """
        z = self.linear_unit.activation(x) # linear computation
        a = self.activation.activation(z) # non-linear activation
        return a

    def add_l2_reg(self, lmbd):
        if lmbd is 0:
            return
        m = self.linear_unit.w.shape[-1]
        self.linear_unit.w += lmbd / m * self.linear_unit.w

    def update_weights(self, learning_rate):
        self.linear_unit.update_weights(learning_rate)

    def backward(self, da):
        """NN backward propagation

        Attributes:
            da (n_x, n_h, n_m) matrix: derivative of activation from current layer (L). AKA dx(L+1)
        Returns:
            dx: AKA da(L-1)
        """
        dz = self.activation.derivative(da)
        da = self.linear_unit.derivative(dz)
        return da


class OutputLayer(NNLayer):
    def __init__(self, n_x, n_h, activation, cost_function):
        NNLayer.__init__(self, n_x, n_h, activation)
        self.cost_function = cost_function

    def cost(self, y):
        a = self.activation.a
        return self.cost_function.cost(y, a)

    def backward(self, y):
        a = self.activation.a
        dc = self.cost_function.cost_d(y, a)
        # same steps as normal NNLayer
        dz = self.activation.derivative(dc)
        da = self.linear_unit.derivative(dz)
        return da


class OutputLayerShortcut(NNLayer):
    def backward_shortcut(self, y):
        a = self.activation.a
        dz = a - y
        self.activation.dz = dz
        da = self.linear_unit.derivative(dz)
        return da


class SoftmaxCategoricalLayer(OutputLayer, OutputLayerShortcut):
    def __init__(self, n_x, n_h):
        OutputLayer.__init__(self, n_x, n_h, activation=Softmax(), cost_function=CategoricalCrossEntropy())


class SigmoidBinaryLayer(OutputLayer, OutputLayerShortcut):
    def __init__(self, n_x, n_h):
        OutputLayer.__init__(self, n_x, n_h, activation=Sigmoid(), cost_function=BinaryCrossEntropy())


class Unit:
    def __init__(self):
        pass

    def activation(self, z):
        raise NotImplementedError  # you want to override this on the child classes

    def derivative(self, z):
        raise NotImplementedError  # you want to override this on the child classes


class LinearUnit(Unit):
    def __init__(self, n_x, n_h):
        Unit.__init__(self)
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
        self.dx, self.dw, self.db = linear_d(dz, self.w, self.x)
        return self.dx

    def update_weights(self, learning_rate):
        self.w -= learning_rate * self.dw
        self.b -= learning_rate * self.db


def initialize_weights(n_x, n_h):
    w = np.random.randn(n_h, n_x) * xavier_initialization(n_x)
    b = np.zeros((n_h, 1), dtype=np.float32)
    return w, b


def xavier_initialization(n_x):
    return 2 / n_x


class ActivationUnit(Unit):
    """Abtract class for RELU, Sigmoid, Softmax activation functions. Combine with Linear Unit

    Attributes:
        a (n_x, n_h) matrix: activation of unit (L)
        dz: derivative of linear unit (L).
            Calculated inside Activation Unit as it depends on a and da
            Type of multiplication depends on the activation - See differences between RELU and Softmax

        !da: (AKA dx) Calculated in the Linear Unit
    """
    def __init__(self):
        Unit.__init__(self)
        self.a = None
        self.dz = None

    def activation(self, z):
        """Activation unit Forward Step
        Args:
            z (n_x, n_h, m): calculation from linear unit (L)
        Returns:
            a: non-linear activation (L)
        """
        raise NotImplementedError  # you want to override this on the child classes

    def derivative(self, da):  # AKA dx of linear layer
        """Activation unit Forward Step
        Args:
            da (n_x, n_h, m): dx of linear layer (L + 1). Derivative of this activation unit
        Returns:
            dz: derivative of current linear unit (L)
        """
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
    """Abtract class for loss functions.

    Attributes:
        c: cost between target Y and predicted A
        dc: derivative of cost function
    """
    def __init__(self):
        self.c = None
        self.dc = None

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
        self.c = categorical_cross_entropy(y, a)
        return self.c

    def cost_d(self, y, a):
        def categorical_cross_entropy_d(_y, _a):
            return - (_y / _a)
        self.dc = categorical_cross_entropy_d(y, a)
        return self.dc


class BinaryCrossEntropy(Cost):
    def cost(self, y, a):
        def binary_cross_entropy(_y, _a):
            cost = _y * np.log(_a) + (1 - _y) * np.log(1 - _a)
            return - np.mean(cost)
        self.c = binary_cross_entropy(y, a)
        return self.c

    def cost_d(self, y, a):
        def binary_cross_entropy_d(_y, _a):
            # cost_d = y / a + (1 - y) / (1 - a)
            cost_d = _y - _a / (_y * (1 - _y))  # same as above
            return - cost_d
        self.dc = binary_cross_entropy_d(y, a)
        return self.dc


def forward_pass(X, Y, nnlayers, lmbd=0):
    hidden_layers = nnlayers[:-1]
    output_layer = nnlayers[-1]

    input_x = X
    for layer in hidden_layers:
        input_x = layer.forward(input_x)

    a = output_layer.forward(input_x)
    cost = output_layer.cost(Y)

    return cost, a


def l2_reg_cost(nnlayers, y, lmbd):
    if lmbd is 0:
        return 0
    m = y.shape[-1]
    weights = get_weights(nnlayers, include_biases=False)
    weights = flat_array(weights)
    cost = lmbd / (m * 2) * np.sum(weights ** 2)
    print('L2 cost:', cost)
    return cost


def backprop_shortcut(Y, nnlayers):
    output_layer, *hidden_layers = reversed(nnlayers)
    output_da = output_layer.backward_shortcut(Y)
    for layer in hidden_layers:
        output_da = layer.backward(output_da)
    return output_da


def backprop(Y, nnlayers):
    output_da = Y
    for layer in reversed(nnlayers):
        output_da = layer.backward(output_da)
    return output_da


def update_weights(nnlayers, learning_rate):
    for layer in nnlayers:
        layer.update_weights(learning_rate)


def add_l2_reg(nnlayers, lmbd):
    for layer in nnlayers:
        layer.add_l2_reg(lmbd)


class Model:
    def __init__(self, X_train, Y_train, X_test, Y_test, nnlayers):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.nnlayers = nnlayers

    def run(self, num_iterations=50, learning_rate=0.01, lmbd=0):
        use_l2_reg = lmbd > 0
        for i in range(num_iterations):
            # forward pass
            cost, _ = forward_pass(self.X_train, self.Y_train, self.nnlayers)
            l2_cost = l2_reg_cost(self.nnlayers, self.Y_train, lmbd)
            cost += l2_cost
            print('Cost:', cost)

            # backprop(Y_train, nnlayers)
            backprop_shortcut(self.Y_train, self.nnlayers)

            # l2 regularization
            add_l2_reg(self.nnlayers, lmbd)

            # update layer weights
            update_weights(self.nnlayers, learning_rate)

        acc = self.get_accuracy()
        print('Accuracy:', acc)

    def get_accuracy(self, x=None, y=None):
        if x is None and y is None:
            x = self.X_test
            y = self.Y_test

        # Accuracy
        cost, a_L = forward_pass(x, y, self.nnlayers, lmbd=0)
        # pred = np.round(a3)

        # this is for cross entropy
        pred = np.zeros(a_L.shape)
        pred[a_L.argmax(axis=0), np.arange(a_L.shape[1])] = 1

        acc = np.mean(pred == y)
        return acc

# Let's create a model with 2 hidden layers with 100 units
def test_model(X_train, Y_train, X_test, Y_test, num_iterations=50, learning_rate=0.01):
    n_x, n_m = X_train.shape
    n_y, _ = Y_train.shape
    # n_y = 1
    n_h1, n_h2 = [100, 100]

    layer1 = NNLayer(n_x, n_h1, activation=RELU())
    layer2 = NNLayer(n_h1, n_h2, activation=RELU())
    layer3 = SoftmaxCategoricalLayer(n_h2, n_y)
    nnlayers = [layer1, layer2, layer3]

    model = Model(X_train, Y_train, X_test, Y_test, nnlayers)
    model.run(num_iterations, learning_rate, .5)


def get_weights(nnlayers, include_biases=True):
    res = []
    for layer in nnlayers:
        res.append(layer.linear_unit.w)
        if include_biases:
            res.append(layer.linear_unit.b)
    return res


def get_gradients(nnlayers):
    res = []
    for layer in nnlayers:
        res.append(layer.linear_unit.dw)
        res.append(layer.linear_unit.db)
    return res


def flat_array(x):
    res = np.array([])
    for arr in x:
        res = np.concatenate((res, arr.flatten()))
    return res


def replace_weights(nnlayers, flat_weights):
    index = 0
    for layer in nnlayers:
        w = layer.linear_unit.w
        w_s = w.size
        w_new = flat_weights[index:index+w_s].reshape(w.shape)
        layer.linear_unit.w = w_new
        index += w_s

        b = layer.linear_unit.b
        b_s = b.size
        b_new = flat_weights[index:index+b_s].reshape(b.shape)
        layer.linear_unit.b = b_new
        index += b_s


def gradient_check(X, Y):
    n_x, n_m = X.shape
    # n_y, _ = Y_train.shape
    n_y = 1
    n_h1, n_h2 = [10, 10]

    layer1 = NNLayer(n_x, n_h1, activation=RELU())
    layer2 = NNLayer(n_h1, n_h2, activation=RELU())
    layer3 = SigmoidBinaryLayer(n_h2, n_y)
    nnlayers = [layer1, layer2, layer3]

    cost, _ = forward_pass(X, Y, nnlayers)
    print('Cost:', cost)
    backprop_shortcut(Y, nnlayers)


    epsilon = .0001
    weights = get_weights(nnlayers)
    unrolled_weights = flat_array(weights)

    approx_gradients = np.empty(unrolled_weights.shape)

    for i in range(unrolled_weights.size):
        thetaplus = copy.deepcopy(unrolled_weights)
        thetaminus = copy.deepcopy(unrolled_weights)

        thetaplus[i] = (thetaplus[i] + epsilon)
        thetaminus[i] = (thetaminus[i] - epsilon)

        replace_weights(nnlayers, thetaplus)
        J_plus, _ = forward_pass(X, Y, nnlayers)

        replace_weights(nnlayers, thetaminus)
        J_minus, _ = forward_pass(X, Y, nnlayers)

        approx = (J_plus - J_minus) / (2 * epsilon)
        approx_gradients[i] = approx

    def euclidean(x):
        return np.sqrt(np.sum(x ** 2))

    np_gradients = flat_array(get_gradients(nnlayers))
    numerator = euclidean(np_gradients - approx_gradients)
    denominator = euclidean(np_gradients) + euclidean(approx_gradients)
    difference = numerator / denominator
    return difference


# Binary class
# (x_train, y_train), (x_test, y_test) = load_data.load_binary_class_data()
# test_model(x_train[:, :100], y_train[:100], x_test[:, :100], y_test[:100])

# gc_error = gradient_check(x_train[:, :100], y_train[:100])
# print('Gradient check error:', gc_error)


# import matplotlib.pyplot as plt
# plt.imshow(x_train[:, 1].reshape(28, 28))

# Categorical class
(x_train, y_train), (x_test, y_test) = load_data.load_class_data(10)
test_model(x_train, y_train, x_test, y_test)
# test_model(x_train[:, :1000], y_train[:, :1000], x_test[:, :1000], y_test[:, :1000])


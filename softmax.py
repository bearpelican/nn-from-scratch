import numpy as np

# Softmax activation function
def softmax(z):
    # Shift z values so highest value is 0
    # Must stabilize as exp can get out of control
    z_norm = z - np.max(z)
    exp = np.exp(z_norm)
    return exp / np.sum(exp, axis=0, keepdims=True)


# Softmax gradient using jacobian
def softmax_grad_simple(s):
    # input s is softmax value of the original input x. Its shape is (1,n)
    # e.i. s = np.array([0.3,0.7]), x = np.array([0,1])

    # make the matrix whose size is n^2.
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else:
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m


# Vectorized version of softmax gradient
def softmax_grad(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)


# Implementation found here: https://github.com/eliben/deep-learning-samples/blob/master/softmax/softmax.py
def softmax_gradient_eliben(z):
    """Computes the gradient of the softmax function.
    z: (T, 1) array of input values where the gradient is computed. T is the
       number of output classes.
    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D


# Alternative softmax solution:
# https://stackoverflow.com/questions/45949141/compute-a-jacobian-matrix-from-scratch-in-python
def softmax_grad_v3(probs):
    n_elements = probs.shape[0]
    jacobian = probs[:, np.newaxis] * (np.eye(n_elements) - probs[np.newaxis, :])
    return jacobian


# (n_class, n_class, n_m_examples)
# Finds softmax for m training examples
def softmax_grad_mexamples(z):
    n_class, n_m = z.shape
    s_grad = np.empty((n_class, n_class, n_m))
    for i in range(z.shape[1]):
        soft_grad = softmax_grad(z[:, i])
        s_grad[:, :, i] = soft_grad
    return s_grad


# Finds softmax gradient for m training examples - transposed version for clarity
def softmax_grad_mexamples_transpose(z):
    # (n_m x n_class) matrix - y class per m row
    n_m, n_class = z.shape
    s_grad = np.empty((n_m, n_class, n_class))
    for i in range(z.shape[0]):
        row = z[i]
        sgr = softmax_grad(row)
        s_grad[i] = sgr
    return s_grad


# x = np.array([1, 2, 3, 4])
#
# sm = softmax(x)
# print(sm)
#
# sm_g_j = softmax_grad(sm)
# print(sm_g_j)
#
#
# sm_g_j = softmax_gradient_eliben(x)
# print(sm_g_j)



z = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [0, 1, 0, 1]]).T
print('Grad m')
sgm2 = softmax_grad_mexamples_transpose(z.T).T
print(sgm2)

sgm = softmax_grad_mexamples(z)
print(sgm)



# dcost_step = binary_cross_entropy_d(Y_train, a3)
# a3_d = sigmoid_d(a3)
# dz3_step = a3_d * dcost_step



'''
cost_d = categorical_cross_entropy_d(Y, a3)
a3_d = softmax_d_m(a3)
print('A3', a3.shape)
print(cost_d.shape)
print(a3_d.shape)
cost_d_r = cost_d.reshape((cost_d.shape[0], 1, cost_d.shape[1]))
dz3_step = np.einsum('ijk,jyk->iyk', a3_d, cost_d_r)
dz3_step_r = dz3_step.reshape((dz3_step.shape[0], dz3_step.shape[2]))

dz3_test = np.einsum('ijk,jk->ik', a3_d, cost_d)

da2, dw3, db3 = linear_d(dz3, w3, a2, b3)
'''
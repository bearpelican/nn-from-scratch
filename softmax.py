import numpy as np

def softmax_grad(s):
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

def softmax_grad_j(softmax):
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def softmax_grad_m(z):
    # Remember to reverse n_m and n_class
    zt = z.T
    n_m, n_class = zt.shape
    s_grad = np.empty((n_m, n_class, n_class))
    for i in range(zt.shape[0]):
        row = zt[i]
        soft_grad = softmax_gradient_eliben(row)
        s_grad[i] = soft_grad
    return s_grad

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

def softmax_grad_j_2(softmax):
    m = softmax.shape[1]
    for i in range(m):
        s = softmax[:, i].reshape(-1,1)
        asdf = np.diagflat(s) - np.dot(s, s.T)
        print(asdf)

def softmax_grad_3(probs):
    n_elements = probs.shape[0]
    jacobian = probs[:, np.newaxis] * (np.eye(n_elements) - probs[np.newaxis, :])
    return jacobian

def softmax(z):
    # Shift z values so highest value is 0
    # Must stabilize as exp can get out of control
    z_norm = z - np.max(z)
    exp = np.exp(z_norm)
    return exp / np.sum(exp, axis=0, keepdims=True)


# x = np.array([[0, 1], [1, 0], [0, 1]])
x = np.array([1, 2, 3, 4])
#
sm = softmax(x)
print(sm)

sm_g_j = softmax_grad_j(sm)
print(sm_g_j)


sm_g_j = softmax_gradient_eliben(x)
print(sm_g_j)
#
# # softmax_grad_j_2(sm)
#
# print('Newest grad 3')
# print(softmax_grad_3(sm))

sm_g = softmax_grad(sm)
print(sm_g)


# sm_g = softmax_grad_j_2(sm)
# print(sm_g)


# x = np.array([[0, 1], [1, 0], [0, 1]])
# print(np.diag(x).shape)
# print(np.diag(x.T).shape)
# print(np.dot(x, x.T).shape)
# print(np.dot(x.T, x).shape)

z = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [0, 1, 0, 1]]).T
print('Grad m')
sgm = softmax_grad_m(z)
print(sgm)

# dcost_step = binary_cross_entropy_d(Y_train, a3)
# a3_d = sigmoid_d(a3)
# dz3_step = a3_d * dcost_step
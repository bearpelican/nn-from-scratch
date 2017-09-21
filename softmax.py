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


x = np.array([[0, 1], [1, 0], [0, 1]])
# x = np.array([1, 2, 3, 4])
#
# sm = softmax(x)
# print(sm)
#
# sm_g_j = softmax_grad_j(sm)
# print(sm_g_j)
#
# # softmax_grad_j_2(sm)
#
# print('Newest grad 3')
# print(softmax_grad_3(sm))

# sm_g = softmax_grad(sm)
# print(sm_g)


x = np.array([[0, 1], [1, 0], [0, 1]])
print(np.diag(x).shape)
print(np.diag(x.T).shape)
print(np.dot(x, x.T).shape)
print(np.dot(x.T, x).shape)
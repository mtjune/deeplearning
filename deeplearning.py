
import numpy as np
import six
import six.moves.cPickle as pickle

# 活性化関数
# ReLU
relu = np.vectorize(lambda x: max(0., x))
d_relu = np.vectorize(lambda x: 1 if x > 0. else 0.)
# 恒等関数
identity = np.vectorize(lambda x: x)
d_identity = np.vectorize(lambda x: 1)


esp = 0.001

# 各層のユニットの数
# input_layer_n:入力層
# hidden_layers_n:最後の層が出力層，あとは隠れ層
input_layer_n = [100]
hidden_layers_n = [30, 100]
f = [relu, identity]
d_f = [d_relu, d_identity]


# 誤差関数
def err(y, t):
    assert y.shape == t.shape

    er = t - y
    norm = (er * er).sum()

    return norm / 2

def d_err(x, t):
    assert x.shape == t.shape

    er =  t - y
    return er

# 重みの値
W = []
B = []


# 初期化
layers_n = input_layer_n + hidden_layers_n
for i in range(0, len(layers_n) - 1):
    W.append(np.random.rand(layers_n[i + 1], layers_n[i]) * 2 - 1)
    B.append(np.random.rand(layers_n[i + 1], 1) * 2 - 1)



def forward(X):

    assert X.shape[0] == W[0].shape[1]

    u = W[0].dot(X) + B[0]
    U = [u]
    z = f[0](u)
    # 各層の処理
    for l in range(1, len(W)):
        u = W[l].dot(z) + B[l]
        U.append(u)
        z = f[l](u)

    return z


def forward_backward(X, T):
    assert X.shape == T.shape
    assert X.shape[0] == W[0].shape[1]

    u = W[0].dot(X) + B[0]
    U = [u]
    z = f[0](u)
    Z = [z]
    # 各層の処理
    for l in range(1, len(W)):
        u = W[l].dot(z) + B[l]
        U.append(u)
        z = f[l](u)
        Z.append(z)


    delta = [None] * len(W)
    delta[-1] = T - Z[-1]
    dW = delta[-1].dot(Z[-2].T())
    W[-1] += - esp * dW

    for l in range(1, len(W) - 1)[::-1]:











if __name__ == '__main__':

    x = np.random.rand(100, 1) * 255

    y = forward(x)

    print(y)

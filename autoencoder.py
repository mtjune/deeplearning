
import numpy as np
import six
import time
import six.moves.cPickle as pickle
from sklearn.datasets import fetch_mldata
import pylab

# 活性化関数
# ReLU
relu = np.vectorize(lambda x: max(0., x))
d_relu = np.vectorize(lambda x: 1. if x > 0. else 0.)
# 恒等関数
identity = np.vectorize(lambda x: x)
d_identity = np.vectorize(lambda x: 1)


# ハイパーパラメータ
esp_w = 0.005
esp_b = 0.001
mom = 0.5
lam = 0.1

# 各層のユニットの数
# input_layer_n:入力層
# hidden_layers_n:最後の層が出力層，あとは隠れ層
input_layer_n = [784]
hidden_layers_n = [100, 784]
f = [relu, identity]
d_f = [d_relu, d_identity]


# 誤差関数
def err(y, t):
    assert y.shape == t.shape

    batchsize = y.shape[1]

    er = t - y
    norm = (er * er).sum()

    return (norm / batchsize)

def d_err(y, t):
    assert y.shape == t.shape

    er =  y - t
    return er

# 重みの値
W = []
B = []


# 初期化
layers_n = input_layer_n + hidden_layers_n
for i in range(0, len(layers_n) - 1):
    W.append(np.random.normal(0, 0.01, (layers_n[i + 1], layers_n[i])))
    B.append(np.random.normal(0, 0.01, (layers_n[i + 1], 1)))

# モメンタムに使う前回の重み修正値
delta_w_p = None


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

    return Z[-1]


def forward_backward(X, T):
    global delta_w_p
    assert X.shape[0] == W[0].shape[1]

    batchsize = X.shape[1]

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

    delta_w = [None] * len(W)

    for l in range(len(W))[::-1]:

        if l == len(W) - 1:
            delta[l] = d_err(Z[-1], T)
        else:
            delta[l] = d_f[l](U[l] * W[l + 1].T.dot(delta[l + 1]))

        if l == 0:
            dW = (delta[l].dot(X.T) / batchsize) - (lam * W[l])
        else:
            dW = (delta[l].dot(Z[l-1].T) / batchsize) - (lam * W[l])
        dB = delta[l].dot(np.ones((batchsize, 1), dtype=np.float64)) / batchsize

        W[l] += -esp_w * dW
        if delta_w_p:
            W[l] += mom * delta_w_p[l]
        B[l] += -esp_b * dB

        delta_w[l] = -esp_w * dW

    delta_w_p = delta_w


    return Z[-1]







if __name__ == '__main__':

    n_epoch = 50
    batchsize = 100

    mnist = fetch_mldata('MNIST original', data_home=".")
    N = len(mnist.data)


    # mnistの平均を求める
    mnist_sum = np.ndarray((784,), dtype=np.float64)
    for mnist_one in mnist.data:
        mnist_sum += mnist_one / 255

    mnist_mean = (mnist_sum / N)

    mnist_mean_batch = np.zeros((784, batchsize), dtype=np.float64)
    for i in range(batchsize):
        mnist_mean_batch[:, i] = mnist_mean


    # pylab.subplot(5, 5, 1)
    # pylab.axis('off')
    # pylab.imshow(mnist.data[0].reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
    # pylab.subplot(5, 5, 2)
    # pylab.axis('off')
    # pylab.imshow(mnist_mean.reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
    # pylab.subplot(5, 5, 3)
    # pylab.axis('off')
    # pylab.imshow((mnist.data[0] - mnist_mean).reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
    # pylab.show()


    for epoch in range(1, n_epoch + 1):

        perm = np.random.permutation(N)
        max_i = N // batchsize

        for i in six.moves.range(0, max_i):
            x_batch = mnist.data[perm[i * batchsize:(i + 1) * batchsize], :] / 255

            # x_input = x_batch.transpose(1, 0).astype(np.float64) - mnist_mean_batch
            x_input = x_batch.transpose(1, 0).astype(np.float64)

            y = forward_backward(x_input, x_input)

            sumerr = err(y, x_input)

            print("{} / {}\t{}".format((epoch - 1) * N + (i + 1) * batchsize, n_epoch * N, sumerr))

        pickle.dump(W, open("W1.dump", "wb"), -1)
        pickle.dump(B, open("B1.dump", "wb"), -1)

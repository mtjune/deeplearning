
import numpy as np
import six
import time
import six.moves.cPickle as pickle
from sklearn.datasets import fetch_mldata
import pylab
import math

# 活性化関数
# ReLU
relu = np.vectorize(lambda x: max(0., x))
d_relu = np.vectorize(lambda x: 1. if x > 0. else 0.)
# 恒等関数
identity = np.vectorize(lambda x: x)
d_identity = np.vectorize(lambda x: 1)

# ソフトマックス関数
def softmax(u):

    z = np.zeros(u.shape, dtype=np.float64)
    for i in range(u.shape[1]):
        sum_exp = 0.

        z_i = np.zeros((u.shape[0],), dtype=np.float64)

        for j in range(u.shape[0]):
            z_i[j] = math.exp(u[j, i])
            sum_exp += z_i[j]

        z[:, i] = z_i / sum_exp

    return z

def d_softmax(u):
    pass
    # 実際使わない

# ハイパーパラメータ
esp_w = 0.1
esp_b = 0.001
mom = 0.5
lam = 0.1

# 各層のユニットの数
# input_layer_n:入力層
# hidden_layers_n:最後の層が出力層，あとは隠れ層
input_layer_n = [784]
hidden_layers_n = [200, 10]
f = [relu, softmax]
d_f = [d_relu, d_softmax]


# 誤差関数
def err(y, t):
    batchsize = y.shape[1]

    er_sum = 0.

    for i in range(batchsize):
        er_sum -= math.log(y[t[i], i])

    return er_sum / batchsize

def d_err(y, t):
    assert y.shape == t.shape

    er =  y - t
    return er


# 学習に使う教師データを作成
def create_correctdata(t):

    batchsize = len(t)

    output_t = np.zeros((hidden_layers_n[-1], batchsize), dtype=np.float64)


    for i in range(batchsize):
        output_t[t[i], i] = 1.


    return output_t




# 正解数を求める関数
def calc_accuracy(y, t):

    batchsize = y.shape[1]

    correct_num = 0

    for i in range(batchsize):
        z = y[:, i]

        correct_num = 0
        maximum = z[0]
        for j in range(1, len(z)):
            if z[j] > maximum:
                maximum = z[j]
                correct_num = j

        if correct_num == int(t[i]):
            correct_num += 1

    return correct_num




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
            delta[l] = d_f[l](U[l]) * W[l + 1].T.dot(delta[l + 1])

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
        if delta_w_p:
            delta_w[l] += mom * delta_w_p[l]

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
            perm_batch = perm[i * batchsize:(i + 1) * batchsize]
            x_batch = mnist.data[perm_batch, :] / 255
            t_batch = mnist.target[perm_batch].astype(np.uint8)

            # x_input = x_batch.transpose(1, 0).astype(np.float64) - mnist_mean_batch
            x_input = x_batch.transpose(1, 0).astype(np.float64)
            # print(t_batch)
            t_input = create_correctdata(t_batch)



            y = forward_backward(x_input, t_input)

            # for i in range(batchsize):
            #     print(y[:, i])

            acc = calc_accuracy(y, t_batch) / batchsize
            loss = err(y, t_batch)

            print("{} / {}\t{}\t{}".format((epoch - 1) * N + (i + 1) * batchsize, n_epoch * N, acc, loss))

        pickle.dump(W, open("classification_W.dump", "wb"), -1)
        pickle.dump(B, open("classification_B.dump", "wb"), -1)

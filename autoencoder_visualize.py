
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



# 重みの値
W = pickle.load(open("W1.dump", "rb"))
B = pickle.load(open("B1.dump", "rb"))



# モメンタムに使う前回の重み修正値
delta_w_p = None


def forward(X):

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

    return Z[-1]




if __name__ == '__main__':

    batchsize = 50

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


    perm = np.random.permutation(N)

    x_batch = mnist.data[perm[0: batchsize], :] / 255

    x_input = x_batch.transpose(1, 0).astype(np.float64)

    y = forward(x_input)


    for i in range(batchsize):
        pylab.subplot(10, 10, 2 * i + 1)
        pylab.axis('off')
        pylab.imshow(x_input[:, i].reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')

        pylab.subplot(10, 10, 2 * i + 2)
        pylab.axis('off')
        pylab.imshow(y[:, i].reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')

    pylab.show()


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

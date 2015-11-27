import numpy as np
import six
import time
import six.moves.cPickle as pickle
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
W = pickle.load(open("W.dump", "rb"))
B = pickle.load(open("B.dump", "rb"))


if __name__ == '__main__':


    hidden_n = hidden_layers_n[0]

    x = np.zeros((hidden_n, hidden_n), dtype=np.float64)

    for i in range(hidden_n):
        x[i, i] = 1.

    z = f[1](W[1].dot(x) + B[1])

    outputs = z.transpose(1, 0)
    for i in range(len(outputs)):
        pylab.subplot(10, 10, i + 1)
        pylab.axis('off')
        pylab.imshow(outputs[i, :].reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')

    pylab.show()

import numpy as np
import six
import time
import six.moves.cPickle as pickle
import pylab

input_layer_n = [784]
hidden_layers_n = [100, 784]

# 重みの値
W = pickle.load(open("W1.dump", "rb"))
B = pickle.load(open("B1.dump", "rb"))


if __name__ == '__main__':


    hidden_n = W[0].shape[0]

    # x = np.zeros((hidden_n, hidden_n), dtype=np.float64)
    #
    # for i in range(hidden_n):
    #     x[i, i] = 1.
    #
    # z = f[1](W[1].dot(x) + B[1])
    #
    #
    #
    # outputs = z.transpose(1, 0)
    # for i in range(len(outputs)):
    #     pylab.subplot(10, 10, i + 1)
    #     pylab.axis('off')
    #     pylab.imshow(outputs[i, :].reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')
    #
    # pylab.show()

    for i in range(hidden_n):
        w = W[0][i, :]

        pylab.subplot(10, 10, i + 1)
        pylab.axis('off')
        pylab.imshow(w.reshape(28, 28), cmap=pylab.cm.gray_r, interpolation='nearest')

    pylab.show()

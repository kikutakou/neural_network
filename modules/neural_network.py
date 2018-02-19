import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_inv(z):
    return z * (1 - z)

def tanh(x):
    p = np.exp(x)
    n = np.exp(-x)
    return (p - n) / (p + n)

def tanh_inv(z):
    return 1 - z ** 2

def none(x):
    return x

def none_inv(z):
    return 1

class NeuralNetwork(object):
    # learning rate
    yita = 0.1

    def __init__(self, n, label2=0, activate=True):
        self.w = np.array([0.3, 0.3])         # initial weight
        self.i = 0                            # iteration
        self.label2 = label2
        
        # switch activate function
        if not activate:
            self.activate = none
            self.activate_inv = none_inv
        elif self.label2 == 0:
            self.activate = sigmoid
            self.activate_inv = sigmoid_inv
        else:
            self.activate = tanh
            self.activate_inv = tanh_inv

    def u(self, x):
        return np.dot(self.w, x)

    def update(self, x, y):
        # forward
        u = self.u(x)
        z = self.activate(u)
        
        # loss
        E = ((z - y) ** 2).mean()

        # backwords
        dEdw = 2 * (z - y) * self.activate_inv(z) * x

        # update weight
        self.w -= self.yita * dEdw.mean(axis=1)

        # test
        if self.label2 == 0:
            y_est = np.round(u)
        else:
            y_est = np.ceil(y)

        # correct ratio
        ratio = (y_est == y).sum() / len(y)

        # print
        self.i += 1
        print("{:3d} updated, w:{}, loss:{:.5f}, ratio:{}".format(self.i, self.w, E, ratio))

        return E

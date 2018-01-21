import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_inv(f):
    return f * (1 - f)

def tanh(x):
    p = np.exp(x)
    n = np.exp(-x)
    return (p - n) / (p + n)

def tanh_inv(f):
    return 1 - f ** 2


class NeuralNetwork(object):
    yita = 0.1

    def __init__(self, n, label2=0):
        self.w = np.array([0.3, 0.3])
        self.i = 0
        
        # switch activate function
        if label2 == 0:
            self.activate = sigmoid
            self.activate_inv = sigmoid_inv
        else:
            self.activate = tanh
            self.activate_inv = tanh_inv

    def update(self, x, y):
        u = np.dot(self.w, x)
        z = self.activate(u)
        
        # backwords
        loss = ((z - y) ** 2).mean()
        dEdw = 2 * (z - y) * self.activate_inv(z) * x

        # update
        self.w -= self.yita * dEdw.mean(axis=1)

        # test
        ratio = (u * y > 0).sum() / len(y)

        self.i += 1
        print(self.i, "updated", self.w, loss, ratio)

        return loss

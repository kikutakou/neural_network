import argparse
import numpy as np


class PlotData(object):

    def __init__(self, n, pattern='diag', y_labels=[0,1], inverse=False):
        self.n = n
        
        # generate random data
        quadrant1 = np.array([np.random.randn(n) + 2, np.random.randn(n) + 2])
        quadrant2 = np.array([np.random.randn(n) - 2, np.random.randn(n) + 2])
        quadrant3 = np.array([np.random.randn(n) - 2, np.random.randn(n) - 2])
        quadrant4 = np.array([np.random.randn(n) + 2, np.random.randn(n) - 2])

        if pattern == 'diag':
            self.x1 = np.hstack([quadrant4])
            self.x2 = np.hstack([quadrant2])
        elif pattern == 'left-right':
            self.x1 = np.hstack([quadrant1, quadrant4])
            self.x2 = np.hstack([quadrant2, quadrant3])
        elif pattern == 'one-others':
            self.x1 = np.hstack([quadrant4])
            self.x2 = np.hstack([quadrant1, quadrant2, quadrant3])
        else:
            exit("unkown pattern: " + pattern)

        self.y1 = np.full(self.n, y_labels[0], dtype=np.int64)
        self.y2 = np.full(self.n, y_labels[1], dtype=np.int64)

        if inverse:
            self.x2 = -self.x2
            self.y2 = self.y2

    def get_x(self):
        return self.x1, self.x2

    def get_y(self):
        return self.y1, self.y2

    def get_train(self):
        x_train = np.hstack([self.x1, self.x2])
        y_train = np.hstack([self.y1, self.y2])
        return x_train, y_train

    def get_square_mean_x(self):
        sqm1 = self.x1.mean(axis=1)
        sqm2 = self.x2.mean(axis=1)
        return sqm1, sqm2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=300)
    parser.add_argument("-d", "--data-pattern", choices=['diag', 'left-right', 'one-others'], default='diag')
    parser.add_argument("-i", "--inverse-data", action="store_true")
    parser.add_argument("-a", "--activation-func", choices=['tanh', 'sigmoid'], default='tanh')
    args = parser.parse_args()

    xdata, ydata = PlotData(args.n).get_train()

    print("".join([ "{}\t{}\t{}\n".format(x0,x1,y) for x0,x1,y in zip(xdata[0],xdata[1],ydata)]), end="")



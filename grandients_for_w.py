import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from modules.neural_network import sigmoid
import modules.pyplot_3d as plt
from modules.load_data import Data
import time


# argparse
parser = argparse.ArgumentParser()
parser.add_argument("data", help="data per line, [x[0],x[1],label] data separated by tab")
args = parser.parse_args()
data = Data(args.data)


# plot
plt.main_ax(xrange=[-5, 5], yrange=[-5, 5])
plt.plot3d(*data.x1)


# mesh for gradients of w
mesh = np.meshgrid(np.arange(-5, 5.1, 0.1), np.arange(-5, 5.1, 0.1))

# dot products for N x MESH
u = sum(np.expand_dims(ww, axis=2) * xx for ww, xx in zip(mesh, data.x1))

# gradients
#Ew = ((u - 1) ** 2).mean(axis=2)           #linear
Ew = ((sigmoid(u) - 1) ** 2).mean(axis=2)       #sigmoid



plt.mesh3d(*mesh, Ew)
plt.show()


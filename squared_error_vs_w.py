# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)


N = 300

#add plot
data = np.array([np.random.randn(N) + 2, np.random.randn(N) * 2])
x,y = data


#mesh w
wx = np.arange(-5, 5.1, 0.1)
wy = np.arange(-5, 5.1, 0.1)
wxy = np.meshgrid(wx, wy)
u = wxy[0] * data[0].reshape((-1,1,1)) + wxy[1] * data[1].reshape((-1,1,1))

#Ew = ((u - 1) ** 2).mean(axis=0)           #linear
Ew = ((sigmoid(u) - 1) ** 2).mean(axis=0)       #sigmoid


ax = fig.add_subplot(111, projection='3d')
ax.plot(data[0], data[1], "wo")
ax.plot_wireframe(wxy[0], wxy[1], Ew)




plt.show()


# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


N = 300

max = 5.
_x,_y = np.meshgrid([-max,max],[-max,max])
_z = np.zeros(_x.shape)


fig = plt.figure()

ax = Axes3D(fig)

#add plot
data1 = np.array([np.random.randn(N) + 1, np.random.randn(N) + 1, np.ones(N)])
ax.scatter3D(data1[0], data1[1], data1[2], color="r")

data2 = np.array([np.random.randn(N) + 3, np.random.randn(N) + 3, np.ones(N)])
ax.scatter3D(data2[0], data2[1], data2[2], color="b")

data2inv = - data2
ax.scatter3D(data2inv[0], data2inv[1], data2inv[2], color="gray")


alldata = data1 + data2inv
print(-alldata.sum(axis=1))




w = np.array([-3,-2,3])
ax.quiver(w[0],w[1],w[2],w[0],w[1],w[2], color="g", length=np.linalg.norm(w))

wz = (w[0] * _x + w[1] * _y) / - w[2]
print(_x)
print(_y)
print(wz)


#all data
ax.plot([-max,max], [0,0], [1,1], color="gray", alpha=0.5)
ax.plot([0,0], [-max,max], [1,1], color="gray", alpha=0.5)
ax.plot([-max,max], [0,0], [0,0], color="k")
ax.plot([0,0], [-max,max], [0,0], color="k")
ax.plot_surface(_x,_y,_z, color="cyan", alpha=0.1)
ax.plot_surface(_x,_y,wz, color="gray", alpha=0.1)
ax.set_xlim(-max,max)
ax.set_ylim(-max,max)
ax.set_zlim(-2,max*2-2)

ax.set_aspect('equal', 'box')



plt.show()


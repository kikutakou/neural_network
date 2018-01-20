# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
x = np.arange(-10,10,0.5)
y = np.arange(-10,10,0.5)


def setax(ax):
    ax.plot([-10,10], [0,0], [0,0], color="k")
    ax.plot([0,0], [-10,10], [0,0], color="k")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')



#vector 1
x,y = np.meshgrid(x,y)
w1 = np.array([2,3])
z1 = w1.dot(np.array([x.ravel(), y.ravel()])).reshape(x.shape)

ax = fig.add_subplot(231, projection='3d')
setax(ax)
ax.set_zlim(-40,40)
ax.plot_wireframe(x,y,z1, color="b")

#vector 2
w2 = np.array([-3,2])
z2 = w2.dot(np.array([x.ravel(), y.ravel()])).reshape(x.shape)

ax = fig.add_subplot(232, projection='3d', sharez=ax)
setax(ax)
ax.set_zlim(-40,40)
ax.plot_wireframe(x,y,z2, color="r")

#sum
ax = fig.add_subplot(233, projection='3d', sharez=ax)
setax(ax)
ax.set_zlim(-40,40)
ax.plot_wireframe(x,y,(z1+z2)/2, color="g")


#vector 1 sigmoid
s1 = 1 / (1 + np.exp(-z1/4))
ax = fig.add_subplot(234, projection='3d')
setax(ax)
ax.set_zlim(0,1)
ax.plot_wireframe(x,y,s1, color="b")

#vector 2 sigmoid
s2 = 1 / (1 + np.exp(-z2/4))
ax = fig.add_subplot(235, projection='3d', sharez=ax)
setax(ax)
ax.set_zlim(0,1)
ax.plot_wireframe(x,y,s2, color="r")

#sum
ax = fig.add_subplot(236, projection='3d', sharez=ax)
setax(ax)
ax.set_zlim(0,1)
ax.plot_wireframe(x,y,(s1*s2)/2, color="g")


fig.tight_layout()

plt.savefig(os.path.splitext(os.path.basename(__file__))[0] + ".pdf")
plt.show()



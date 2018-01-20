# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time

def get_ax_size(ax1):
    bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

def axsize(width=1.0, height=1.0, left=0.0, top=0.0, wmargin=0.1, hmargin=0.1):
    l = left + width * wmargin
    b = (1. - top - height) + height * hmargin
    w = width * (1. - 2. * wmargin)
    h = height * (1. - 2. * hmargin)
    return [l,b,w,h]



N = 300



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    p = np.exp(x)
    n = np.exp(-x)
    return (p - n) / (p + n)



fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(111, projection='3d')

#add plot
data1 = np.array([np.random.randn(N) + 2, np.random.randn(N) + 2])
data2 = np.array([np.random.randn(N) - 2, np.random.randn(N) - 2])
data = np.hstack([data1, -data2]).T

x = np.arange(-10,10,0.5)
y = np.arange(-10,10,0.5)
x,y = np.meshgrid(x,y)

xy = np.stack([x,y])
z = [d[0] * xy[0] + d[1] * xy[1] for d in data]
z = sum(z) / len(z)
z = tanh(z)





ax1.plot(data1[0], data1[1], "bo")
ax1.plot(data2[0], data2[1], "ro")
ax1.set_xlim(-5,5)
ax1.set_ylim(-5,5)


ax1.plot_wireframe(x,y,z, color="g")

plt.show()




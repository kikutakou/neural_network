# -*- coding: utf-8 -*-
import os
import math
import numpy as np
from modules.neural_network import sigmoid
import modules.pyplot_3d_compare as plt
from mpl_toolkits.mplot3d import Axes3D

# x and y
x = np.arange(-10,10,0.5)
y = np.arange(-10,10,0.5)
x, y = np.meshgrid(x,y)




# vector 1
w1 = np.array([2,3])
z1 = w1[0] * x + w1[1] * y
ax = plt.plot_new(x, y, z1, color="b", zrange=(-40, 40))

# vector 2
w2 = np.array([-3,2])
z2 = w2[0] * x + w2[1] * y
ax = plt.plot_new(x, y, z2, color="r", zrange=(-40, 40))

# sum
z3 = (z1+z2)/2
ax = plt.plot_new(x, y, z3, color="g", zrange=(-40, 40))



# vector 1 sigmoid
s1 = sigmoid(z1)
ax = plt.plot_new(x, y, s1, color="b", zrange=(0, 1))

# vector 2 sigmoid
s2 = sigmoid(z2)
ax = plt.plot_new(x, y, s2, color="r", zrange=(0, 1))

# sum of sigmoid
s3 = (s1 * s2) / 2
ax = plt.plot_new(x, y, s3, color="g", zrange=(0, 1))



plt.save_to(os.path.splitext(os.path.basename(__file__))[0] + ".pdf")
plt.show()



#/usr/bin/python

import numpy as np
import random
import matplotlib.pyplot as plt
import os

def bc2xy(bc):
  y = bc[1] * np.sqrt(3.0) / 2.0
  w = 1.0 - bc[1]
  x = y / np.sqrt(3.0) + np.where(bc[1] == 1.0, bc[0], w * bc[0] / (1.0 - bc[1]))
  return x,y


### triangle plot
plt.figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
plt.ylim(0, 0.9)
plt.xlim(0, 1.0)

# edges for triangle
edge = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0], [0.0, 0.0]])
plt.plot(edge[:,0], edge[:,1])


#data
abc = np.array([1, 2, 2])


def softmax(abc):
  abc_exp = np.exp(abc)
  return abc_exp / np.sum(abc_exp)



plt.plot(*bc2xy(softmax(abc)[1:3]), 'o')
plt.show()



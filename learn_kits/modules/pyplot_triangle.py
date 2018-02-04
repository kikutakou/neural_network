# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.figure(num=None, figsize=(5, 5), dpi=80, facecolor='w', edgecolor='k')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.tick_params(labelbottom="off",bottom="off")
plt.tick_params(labelleft="off",left="off")
plt.xticks([])
plt.yticks([])
plt.axis('off')

# edges for triangle
bottom_left = [0.0, 0.0]
bottom_right = [1.0, 0.0]
top = [0.5, np.sqrt(3.0) / 2.0]
edge = np.array([bottom_left, bottom_right, top, bottom_left])
plt.plot(edge[:,0], edge[:,1], "k")
height = np.sqrt(3.0) / 2.0

def bc2xy(data):
    if len(data) > 3 or len(data) < 2:
        raise ValueError("data can only contain 3 or 2 elements")
    elif len(data) == 3 and sum(data) != 1.0:
        raise ValueError("given 3 elements, the sum must be 1.0")
    elif sum(data) > 1.0:
        raise ValueError("given 2 elements, the sum must be less than 1.0")

    y = data[0] * height
    shift = data[0] / 2.0
    x = shift + 1.0 - data[0] - data[1]
    return x,y


def plot(data):
    plt.plot(*bc2xy(data), 'o')

def show():
    plt.show()


if __name__ == '__main__':
    
    add_main_ax()
    plt.show()

import argparse
import random
import numpy as np
import sys
import matplotlib.pyplot as plt



fig = plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')

i = 1
ax = None
def plot_new(x, y, z, color="b", zrange=(0,1)):
    global i
    global ax
    ax = fig.add_subplot(230 + i, projection='3d')
    ax.set_zlim(*zrange)
    
    # draw base axis
    ax.plot([-10,10], [0,0], [0,0], color="k")
    ax.plot([0,0], [-10,10], [0,0], color="k")

    # label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # draw wires
    ax.plot_wireframe(x, y, z, color=color)
    
    i += 1
    return ax



def show():
    fig.tight_layout()
    plt.show()

def save_to(file):
    plt.savefig(file)


if __name__ == '__main__':

    plt.show()

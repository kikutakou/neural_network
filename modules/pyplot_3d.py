import argparse
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure(figsize=(7,7))


ax = fig.add_subplot(111, projection='3d')


def plot3d(*data, type="o", xrange=[-5, 5], yrange=[-5, 5]):
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    ax.plot(data[0], data[1], "wo")

def mesh3d(x, y, z):
    ax.plot_wireframe(x,y,z)



def show():
    plt.show()



if __name__ == '__main__':

    plt.show()

# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# setup colors
#colors = ('red', 'blue', 'green')
x = np.arange(10)
ys = [i+x+(i*x)**2 for i in range(10)]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

width = 0.3
labels = []




def plot(data, smax, norm=False):

    labels.append(str(data))
    n = len(labels)    # number of data
    
    if norm:
        # normalize to [0, 1]
        data = data / data.sum()
    else:
        # normalize to data.sum()
        smax *= data.sum()

    # data bar
    bottom = 0.0
    for i, d in enumerate(data):
        plt.bar(n - width - 0.05, d, width, color=colors[i], bottom=bottom)
        bottom += d
    
    # softmax bar
    bottom = 0.0
    for i, d in enumerate(smax):
        plt.bar(n + 0.05, d, width, color=colors[i], bottom=bottom)
        bottom += d



def show():
    plt.xticks(1 + np.arange(len(labels)), labels)
    plt.xlim(0, len(labels) + 1)
    plt.show()


#
#N = 5
#data = np.array([[20, 25], [35, 32], [30, 34], [35, 20], [27, 25]])
#
#menMeans = (20, 35, 30, 35, 27)
#womenMeans = (25, 32, 34, 20, 25)
#ind = np.arange(data.shape[0]) + 0.5    # the x locations for the groups
#
#p1 = plt.bar(ind, data[:, 0], width, color='red')
#p2 = plt.bar(ind, data[:, 1], width, bottom=data[:, 0])
#plt.xticks(ind + width / 2.0, ('G1', 'G2', 'G3', 'G4', 'G5'))
#
#plt.legend((p1[0], p2[0]), ('Men', 'Women'))
#
#plt.xlim(0, 5)
#plt.show()


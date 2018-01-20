# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

N = 300

SHOW_Z = True

DATA_DIAGONAL = True
DATA_LR = True

BATCH = False

INVERSE = False

SIGMOID = False
TANH = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    p = np.exp(x)
    n = np.exp(-x)
    return (p - n) / (p + n)

if SHOW_Z:
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_axes(axsize(1.0,0.8))
    #print(get_ax_size(ax1))
    #exit(1)
else:
    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(1,1,1)


#add plot

data1 = np.array([np.random.randn(N) + 2, np.random.randn(N) + 2])
data2 = np.array([np.random.randn(N) - 2, np.random.randn(N) + 2])
data3 = np.array([np.random.randn(N) - 2, np.random.randn(N) - 2])
data4 = np.array([np.random.randn(N) + 2, np.random.randn(N) - 2])

if DATA_DIAGONAL:
    x_data1 = np.hstack([data4])
    x_data2 = np.hstack([data2])
elif DATA_LR:
    x_data1 = np.hstack([data1,data4])
    x_data2 = np.hstack([data2,data3])
else:
    x_data1 = np.hstack([data4])
    x_data2 = np.hstack([data1,data2,data3])

y_data1 = np.ones(len(x_data1[0])).reshape([1,-1])
y_data2 = -np.ones(len(x_data2[0])).reshape([1,-1])

if INVERSE:
    x_data2, y_data2 = -x_data2, -y_data2

#prepare data, mixing by randam swap
x_train = np.hstack([x_data1,x_data2])
y_train = np.hstack([y_data1,y_data2])
for i in range(3 * len(x_train[0])):
    p,q = np.random.randint(0, len(x_train[0]), 2)
    if p != q:
        x_tmp,y_tmp = np.copy(x_train[:,p]),np.copy(y_train[:,p])
        x_train[:,p],y_train[:,p] = x_train[:,q],y_train[:,q]
        x_train[:,q],y_train[:,q] = x_tmp,y_tmp
#x_train = np.vstack([x_data1,x_data2]).T.reshape(-1,2).T
#y_train = np.vstack([y_data1,y_data2]).T.reshape(-1,1).T

#optimal solution for square mean
m = (x_train * x_train.reshape((2,1,-1))).sum(axis=2)
v = (x_train * y_train).sum(axis=1)
opt = np.dot(np.linalg.inv(m),v)
plt.quiver(opt[0],opt[1], angles='xy',scale_units='xy',scale=0.2, color="pink")

if SIGMOID:
    y_train, y_data2 = y_train * (y_train > 0), y_data2 * (y_data2 > 0)


ax1.plot(x_data1[0], x_data1[1], "bo")
ax1.plot(x_data2[0], x_data2[1], "ro")
ax1.set_xlim(-5,5)
ax1.set_ylim(-5,5)

if SHOW_Z:
    ax2 = fig.add_axes(axsize(1.0,0.2,0,0.8), sharex=ax1)
    x2 = np.arange(-5, 5, 0.1)
    y2 = x2
    if SIGMOID:
        y2 = sigmoid(x2)
    elif TANH:
        y2 = tanh(x2)
    else:
        y2 = x2

#parameter : yita = speed, w = initial weight
yita = 0.01
w = np.array([0.3, 0.3])



def _update_plot(i, ax, im, w):

    #remove previous fig
    while len(im) > 0:
        im.pop().remove()


    if BATCH:
        #list of vector
        x = x_train
        y = y_train
    else:
        #single vector
        x = x_train[:,i]
        y = y_train[:,i]

    u = np.dot(w,x)

    if SIGMOID:
        f = sigmoid(u)
        dE = (f - y) * (1 - f) * f * x
    elif TANH:
        f = tanh(u)
        dE = (f - y) * (1 - f * f) * x
    else:
        dE = (u - y) * x

    if BATCH:
        dE = dE.mean(axis=1)     #vector to scalor

    w[0],w[1] = (tuple(w - yita * dE))
    y_res = np.dot(w,x_train) > 0
    ratio = (y_res == (y_train > 0)).sum()

    print("updated", y[0]>0, w, ratio)
    im.append(ax[0].quiver(w[0],w[1], angles='xy',scale_units='xy',scale=0.2, color="magenta"))

    #axis 2
    if SHOW_Z:
        ax[1].cla()
        z_data1 = np.dot(w, x_data1)
        z_data2 = np.dot(w, x_data2)
        ax2.plot(z_data1, y_data1[0], "bo")
        ax2.plot(z_data2, y_data2[0], "ro")
        ax2.plot(x2, y2, "k:")
        if SIGMOID:
            ax[1].set_ylim(-0.5,1.5)
        else:
            ax[1].set_ylim(-1.5,1.5)






# build animation
im = [] # array to keep frames
if SHOW_Z:
    ax = [ax1, ax2]
else:
    ax = [ax1]

ani = animation.FuncAnimation(fig, _update_plot, fargs = (ax, im, w), frames=x_train[0].size, interval=1, repeat=False)

#FFwriter = animation.FFMpegWriter()
#ani.save('simple_perseptron.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#ani.save('simple_perseptron.gif', writer='imagemagick')
#print('saved')

if not SHOW_Z:
    plt.gca().set_aspect('equal', adjustable='box')
plt.show()




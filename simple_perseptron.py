# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from modules.load_data import Data
from modules.neural_network import NeuralNetwork


def axsize(width=1.0, height=1.0, left=0.0, top=0.0, wmargin=0.1, hmargin=0.1):
    l = left + width * wmargin
    b = (1. - top - height) + height * hmargin
    w = width * (1. - 2. * wmargin)
    h = height * (1. - 2. * hmargin)
    return [l,b,w,h]

def add_main_axis(zview=True):
    if args.add_z_view:
        fig = plt.figure(figsize=(7,8))
        fig.add_axes(axsize(1.0,0.8))
    else:
        fig = plt.figure(figsize=(7,7))
        fig.add_subplot(1,1,1)

def add_sub_axis():
    plt.gcf().add_axes(axsize(1.0,0.2,0,0.8), sharex=plt.gcf().axes[0])


def add_plots_main(x1, x2):
    ax1 = plt.gcf().axes[0]
    ax1.plot(*x1, "o")
    ax1.plot(*x2, "o")
    ax1.set_xlim(-5,5)
    ax1.set_ylim(-5,5)


def add_mean_main(sqm):
    ax1 = plt.gcf().axes[0]
    ax1.quiver(*sqm, angles='xy',scale_units='xy',scale=1.0, color="pink")


def plot_w_main(w):
    ax1 = plt.gcf().axes[0]
    quiv = ax1.quiver(*w, angles='xy',scale_units='xy',scale=1.0, color="magenta")
    return quiv


afunc = None
def set_sub(activate_func):
    global afunc
    x_afunc = np.arange(-5, 5, 0.1)
    y_afunc = nn.activate(x_afunc)
    afunc = [x_afunc, y_afunc]


def plot_sub(s1, s2):
    ax2 = plt.gcf().axes[1]
    ax2.cla()
    ax2.plot(*s1, "o")
    ax2.plot(*s2, "o")
    ax2.plot(*afunc, "k:")
    ax2.set_ylim(-1.5,1.5)
    ax2.set_xlim(-5,5)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="data per line, [x[0],x[1],label] data separated by tab")
    parser.add_argument("-a", "--activation-func", choices=['tanh', 'sigmoid'], default='tanh')
    parser.add_argument("-s", "--sigmoid", action="store_true")
    parser.add_argument("-z", "--add-z-view", action="store_true")
    args = parser.parse_args()

    if args.sigmoid:
        args.activation_func = 'sigmoid'

    data = Data(args.data)

    # add axis
    add_main_axis(args.add_z_view)

    # set box size
    add_plots_main(data.x1, data.x2)
    add_mean_main(data.x1.mean(axis=1))

    #parameter: yita = learning ratio, w = initial weight
    nn = NeuralNetwork(2, data.y_all.min())
    frames = [plot_w_main(nn.w)]

    if args.add_z_view:
        add_sub_axis()
        set_sub(nn.activate)
        s1 = [np.dot(nn.w, data.x1), data.y1]
        s2 = [np.dot(nn.w, data.x2), data.y2]
        plot_sub(s1, s2)

    # this is animation
    def update_plot(i, nn, frames):
        nn.update(data.x_all, data.y_all)

        # update fig
        frames and frames.pop().remove()
        frames.append(plot_w_main(nn.w))

        # render all plots on axis 2
        if args.add_z_view:
            s1 = [np.dot(nn.w, data.x1), data.y1]
            s2 = [np.dot(nn.w, data.x2), data.y2]
            plot_sub(s1, s2)

    # build animation
    ani = animation.FuncAnimation(plt.gcf(), update_plot, fargs=[nn, frames], frames=1000, interval=1, repeat=False)

    # plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    #FFwriter = animation.FFMpegWriter()
    #ani.save('simple_perseptron.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
    #ani.save('simple_perseptron.gif', writer='imagemagick')
    #print('saved')

    plt.show()






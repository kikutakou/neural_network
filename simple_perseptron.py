# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import sys
import matplotlib.animation as animation
from modules.load_data import Data
from modules.neural_network import NeuralNetwork
import modules.pyplot_2d as plt




activate_plot = None
def update_sub(nn, data):
    global activate_plot
    if activate_plot is None:
        x_afunc = np.arange(-5, 5, 0.1)
        y_afunc = nn.activate(x_afunc)
        activate_plot = [x_afunc, y_afunc]
    s1 = [np.dot(nn.w, data.x1), data.y1]
    s2 = [np.dot(nn.w, data.x2), data.y2]
    plt.clear_sub()
    plt.plot_sub(s1, s2)
    plt.plot_sub(activate_plot, type="k:")



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="data per line, [x[0],x[1],label] data separated by tab")
    parser.add_argument("-z", "--zview", action="store_true")
    parser.add_argument("-e", "--epoch", type=int, default=1000)
    args = parser.parse_args()

    data = Data(args.data)

    # add axis
    if args.zview:
        plt.add_sub_ax()


    #parameter: yita = learning ratio, w = initial weight
    nn = NeuralNetwork(2, data.y_all.min())
    frames = [plt.quiver_main(nn.w, color="magenta")]

    # set initial plot
    plt.plot_main(data.x1, data.x2)
    plt.quiver_main(data.x1.mean(axis=1), color="pink")
    if args.zview:
        update_sub(nn, data)

    # this is animation
    def update_plot(i, nn, frames):
        nn.update(data.x_all, data.y_all)

        # update fig
        while frames:
            frames.pop().remove()
        frames.append(plt.quiver_main(nn.w, color="magenta"))

        # render all plots on axis 2
        if args.zview:
            update_sub(nn, data)

    # build animation
    ani = animation.FuncAnimation(plt.fig, update_plot, fargs=[nn, frames], frames=args.epoch, interval=1, repeat=False)

    # plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    #FFwriter = animation.FFMpegWriter()
    #ani.save('simple_perseptron.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
    #ani.save('simple_perseptron.gif', writer='imagemagick')
    #print('saved')

    plt.show()






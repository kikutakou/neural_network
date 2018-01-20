# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

N = 200
K = 4       #less than 5

COLOR = ["blue", "red", "green", "black"]
PLOT = ["bo", "ro", "go", "ko"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

#add plot
x_data = [np.array([np.random.randn(N) + 2, np.random.randn(N) + 2]),
          np.array([np.random.randn(N) - 2, np.random.randn(N) + 2]),
          np.array([np.random.randn(N) + 2, np.random.randn(N) - 2]),
          np.array([np.random.randn(N) - 2, np.random.randn(N) - 2])]


for k in range(K):
    ax.plot(x_data[k][0], x_data[k][1], PLOT[k])



#mix
x_train = np.vstack(x_data).T.reshape(-1,2).T
y_train = np.tile(np.identity(K),(N,1)).T



#parameter : yita = speed, w = initial weight
yita = 0.01
w = np.tile([0.3, 0.3], (4,1))



def _update_plot(i, fig, im, w):
    rad = math.radians(i)
    
    #remove previous fig
    while len(im) > 0:
        im.pop().remove()
    
    x = x_train[:,i]
    y = y_train[:,i]

    #vector calcuration
    u = np.dot(w,x)
    f = sigmoid(u)
    dEdu = (f - y) * (1 - f) * f

    #(to replace the contents of w)
    for k in range(K):
        dEdw = dEdu[k] * x
        w[k] = w[k] - yita * dEdw
        print("updated", i, k, w[k])
        im.append(plt.quiver(w[k][0],w[k][1], angles='xy',scale_units='xy',scale=0.2, color=COLOR[k]))


# build animation
im = [] # array to keep frames
ani = animation.FuncAnimation(fig, _update_plot, fargs=(fig, im, w), frames=x_train[0].size, interval=1, repeat=False)

#FFwriter = animation.FFMpegWriter()
#ani.save('simple_perseptron.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#ani.save('simple_perseptron.gif', writer='imagemagick')
#print('saved')

plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(os.path.splitext(os.path.basename(__file__))[0] + ".pdf")
plt.show()




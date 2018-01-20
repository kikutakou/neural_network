# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

N = 300
K = 2

COLOR = ["blue", "red", "green", "black"]
PLOT = ["bo", "ro", "go", "ko"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    p = np.exp(x)
    n = np.exp(-x)
    return (p - n) / (p + n)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

#add plot

data1 = np.array([np.random.randn(N) + 2, np.random.randn(N) + 2])
data2 = np.array([np.random.randn(N) - 2, np.random.randn(N) + 2])
data3 = np.array([np.random.randn(N) - 2, np.random.randn(N) - 2])
data4 = np.array([np.random.randn(N) + 2, np.random.randn(N) - 2])

#x_data1,x_data2 = np.hstack([data4]),np.hstack([data2])
#x_data1,x_data2 = np.hstack([data1,data4]),np.hstack([data2,data3])
x_data1 = np.hstack([data4])
x_data2 = np.hstack([data1,data2,data3])
y_data1 = np.ones(len(x_data1[0])).reshape([1,-1])
y_data2 = np.zeros(len(x_data2[0])).reshape([1,-1])

ax.plot(x_data1[0], x_data1[1], "bo")
ax.plot(x_data2[0], x_data2[1], "ro")


#mix
x_train = np.hstack([x_data1,x_data2])
y_train = np.hstack([y_data1,y_data2])
for i in range(3 * len(x_train[0])):
    p,q = np.random.randint(0, len(x_train[0]), 2)
    if p != q:
        x_tmp,y_tmp = np.copy(x_train[:,p]),np.copy(y_train[:,p])
        x_train[:,p],y_train[:,p] = x_train[:,q],y_train[:,q]
        x_train[:,q],y_train[:,q] = x_tmp,y_tmp

#debug
#x_train = np.tile(x_data1a,(1,4))
#y_train = np.tile(np.array([[1,0]]).T, (1,4*N))



#parameter : yita = speed, w = initial weight
yita = 0.01
w2 = np.array([[1., 0.], [0., 1.]])
w3 = np.array([0.3, 0.3])
w = [w2,w3]



def _update_plot(i, fig, im, w):
    rad = math.radians(i)
    
    #remove previous fig
    while len(im) > 0:
        im.pop().remove()
    
    x = x_train[:,i]
    y = y_train[:,i]
    w2 = w[0]
    w3 = w[1]

    u2 = np.dot(w2, x)
    print(u2)
    f2 = tanh(u2)
    print(f2)
    
    u3 = np.dot(w3, f2)
    f3 = sigmoid(u3)
    dEdu3 = (f3 - y) * (1 - f3) * f3

    dEdw3 = dEdu3 * f2
    w3[0] = w3[0] - yita * dEdw3[0]
    w3[1] = w3[1] - yita * dEdw3[1]


    y_res = np.dot(w3,np.dot(w2,x_train)) > 0
    ratio = (y_res == (y_train > 0)).sum()
    print(ratio)
#ratio = ((np.dot(w,x_train) > 0) == (y_train > 0)).sum()


    print("updated", i, w3)
    im.append(plt.quiver(w3[0],w3[1], angles='xy',scale_units='xy',scale=0.2, color="cyan"))



# build animation
im = [] # array to keep frames
ani = animation.FuncAnimation(fig, _update_plot, fargs=(fig, im, w), frames=x_train[0].size, interval=1, repeat=False)

#FFwriter = animation.FFMpegWriter()
#ani.save('simple_perseptron.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
#ani.save('simple_perseptron.gif', writer='imagemagick')
#print('saved')

plt.gca().set_aspect('equal', adjustable='box')
plt.show()




# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


N = 300


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

#add plot
data1 = np.array([np.random.randn(N) + 2, np.random.randn(N) * 2])
data2 = np.array([np.random.randn(N) - 2, np.random.randn(N) * 2])

ax.plot(data1[0], data1[1], "bo")
ax.plot(data2[0], data2[1], "ro")

data2inv = - data2
#ax.plot(data2inv[0], data2inv[1], "go")



training = np.hstack([data1.T,data2inv.T]).reshape(-1,2).T


yita = 0.01
w = np.array([3., 3.])


def _update_plot(i, fig, im, w):
    rad = math.radians(i)
    
    #remove previous fig
    if len(im) > 0:
        im[0].remove()
        im.pop()
    
    #get training data[i]
    t = training[:,i]
    f = np.dot(w,t)
    if f < 0:
        w[0],w[1] = (tuple(w + t * yita))
        print("updated", w)
    im.append(plt.quiver(w[0],w[1], angles='xy',scale_units='xy',scale=1, color="gray"))



# アニメーション作成
im = [] # フレーム更新の際に前回のプロットを削除するために用意
ani = animation.FuncAnimation(fig, _update_plot, fargs = (fig, im, w), frames=training[0].size, interval=1, repeat=False)


plt.gca().set_aspect('equal', adjustable='box')

plt.show()


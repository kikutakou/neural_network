# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def add_main_ax(with_sub=True):
    global ax1
    global fig
    if with_sub:
        fig = plt.figure(figsize=(7,8))
        ax1 = fig.add_axes(axsize(1.0,0.8))
    else:
        fig = plt.figure(figsize=(7,7))
        ax1 = fig.add_subplot(1,1,1)
plt.add_main_ax = add_main_ax



def axsize(width=1.0, height=1.0, left=0.0, top=0.0, wmargin=0.1, hmargin=0.1):
    l = left + width * wmargin
    b = (1. - top - height) + height * hmargin
    w = width * (1. - 2. * wmargin)
    h = height * (1. - 2. * hmargin)
    return [l,b,w,h]

def add_sub_ax():
    global ax2
    ax2 = fig.add_axes(axsize(1.0, 0.2, 0, 0.8), sharex=ax1)
plt.add_sub_ax = add_sub_ax



def plot_main(*data, type="o", xrange=[-5, 5], yrange=[-5, 5]):
    for d in data:
        ax1.plot(*d, type)
    ax1.set_xlim(*xrange)
    ax1.set_ylim(*yrange)
plt.plot_main = plot_main


def quiver_main(data, color="pink"):
    return ax1.quiver(*data, angles='xy',scale_units='xy',scale=1.0, color=color)
plt.quiver_main = quiver_main


def clear_sub():
    ax2.cla()
plt.clear_sub = clear_sub



def plot_sub(*data, type="o", xrange=[-5, 5], yrange=[-1.5, 1.5]):
    for d in data:
        ax2.plot(*d, type)
    ax2.set_xlim(*xrange)
    ax2.set_ylim(*yrange)
plt.plot_sub = plot_sub



if __name__ == '__main__':
    
    add_main_ax()
    plt.show()

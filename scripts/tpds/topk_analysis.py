# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

fig, ax = plt.subplots(1,1,figsize=(5,3.8))

def read_log(fn):
    data = {}
    with open(fn, 'r') as f:
        for line in f.readlines():
            n, t = line.split(',')
            n = int(n)
            t = float(t[0:])
            if n not in data:
                data[n] = []
            data[n].append(t)
    return data

def plot(fn):
    data = read_log(fn)
    print(data)
    ns = data.keys()
    ns.sort()
    ts = []
    for n in ns:
        ts.append(np.mean(data[n]))
    ax.plot(ns, ts)
    #ax.set_xscale("log", basex=2, nonposx='clip')
    plt.show()


if __name__ == '__main__':
    logfile='./logs/topk-k80.log'
    plot(logfile)

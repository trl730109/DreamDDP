# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

def read_topkperf_log(fn):
    f = open(fn)
    print('fn: ', fn)
    sizes = []
    times = []
    for line in f.readlines():
        items = line.split(',')
        size = int(items[0])
        t = float(items[1][:-1])
        sizes.append(size)
        times.append(t)
    f.close()
    return np.array(sizes)[20:400], np.array(times)[20:400]


def read_layerwise_times(fn):
    print('fn: ', fn)
    f = open(fn)
    sizes = []
    times = []
    for line in f.readlines():
        if line.find('layerwise backward times: ') > 0:
            times = line.split('layerwise backward times: ')[-1][1:-2].split(', ')
        if line.find('layerwise backward sizes: ') > 0:
            sizes = line.split('layerwise backward sizes: ')[-1][1:-2].split(', ')
    times = [float(t) for t in times]
    sizes = [int(s) for s in sizes]
    return np.array(sizes), np.array(times)

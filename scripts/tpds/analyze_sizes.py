# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
from communication import read_times_from_nccl_log
from plot_loss2 import STANDARD_TITLES, OUTPUTPATH
import numpy as np

DNN_MARKERS = {
        'resnet152':'d',
        'densenet161':'*',
        'densenet201':'o',
        'inceptionv4':'^',
        }

DNN_COLORS = {
        'resnet152':'green',
        'densenet161':'red',
        'densenet201':'m',
        'inceptionv4':'black',
        }

def read_tensor_sizes(fn):
    print('fn: ', fn)
    with open(fn) as f:
        for l in f.readlines():
            if l.find('layerwise backward sizes: [') > 0:
                str_sizes = l.split('layerwise backward sizes: [')[-1][:-2]
                sizes = str_sizes.split(', ')
                sizes = [int(s) for s in sizes]
                return sizes

def analyze_tensor_sizes():
    fig, ax = plt.subplots(figsize=(6,4.5))
    ax2 = ax.twinx()

    def _plot_dnn_tensor(dnn, bs):
        fn = './logs/allreduce-gwarmup-dc1-model-tpds56GbIB-v2-ada-thres-0kbytes/%s-n16-bs%d-lr0.8000-ns1-ds1.0/hsw224-0.log' % (dnn,bs)
        sizes = read_tensor_sizes(fn)
        counter_dict = {}
        for s in sizes:
            if s not in counter_dict:
                counter_dict[s] = 0
            counter_dict[s] += 1
        keys = list(counter_dict.keys())
        keys.sort()
        print(dnn, 'sizes: ', keys)
        x_pos = [i for i, _ in enumerate(keys)]
        counters = [counter_dict[k] for k in keys]
        print(dnn, 'counters: ', counters)
        print(dnn, 'Total tensors: ', np.sum(counters))
        #ax2.bar(x_pos, counters, color='green')
        ax2.scatter(np.array(keys)*4, counters, color=DNN_COLORS[dnn], marker=DNN_MARKERS[dnn], facecolors='none', linewidth=1, label=STANDARD_TITLES[dnn])
        #ax2.set_xticks(x_pos, keys)
        #ax2.set_xlabel('Tensor size')
        ax2.set_ylabel('Count')
        threshold = 128
        idx = 0
        for i, s in enumerate(keys):
            if s > threshold:
                idx = i
                break
        thres_count = np.sum(counters[0:idx])
        print(dnn, 'counter smaller than threshold: ', thres_count)
    #dnn='densenet201';bs=64
    lines = []
    labels = []
    dnn='resnet152';bs=128
    _plot_dnn_tensor(dnn, bs)
    dnn='densenet161';bs=64
    _plot_dnn_tensor(dnn, bs)
    dnn='densenet201';bs=64
    _plot_dnn_tensor(dnn, bs)
    dnn='inceptionv4';bs=128
    _plot_dnn_tensor(dnn, bs)
    lines, labels = ax2.get_legend_handles_labels()

    mode = 'allreduce'
    comm_fn = './logs/v100-10GbE-allreduce-nccl-small-v6-16.log'
    #comm_fn = './logs/v100-56GbIB-allreduce-nccl-small-v2-16.log'
    comm_sizes, comms, errors = read_times_from_nccl_log(comm_fn, mode=mode)
    #print('min size: ', np.min(comm_sizes), ', max size: ', np.max(comm_sizes))
    #print('min tensor size: ', np.min(keys)*4, ', max tensor size: ', np.max(keys)*4)

    ax.plot(comm_sizes, comms*1e6, label='all-reduce (10GbE)', linewidth=0.5)

    #comm_fn2 = './logs/v100-10GbE-allreduce-nccl-small-v6-16.log'
    comm_fn2 = './logs/v100-56GbIB-allreduce-nccl-small2-16.log'
    comm_sizes2, comms2, errors = read_times_from_nccl_log(comm_fn2, mode=mode)
    ax.plot(comm_sizes2, comms2*1e6, label='all-reduce (56GbIB)', linewidth=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    fig.legend(lines1+lines, labels1+labels, loc='upper center', ncol=3)
    ax.grid(linestyle=':')

    ax.set_xlabel('Size of parameters [bytes]')
    ax.set_ylabel(r'Communication time [$\mu s$]')
    #plt.legend(ncol=1, loc=2, prop={'size': 14})
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xscale('log')
    #plt.title(dnn)
    plt.savefig('%s/%s.pdf' % (OUTPUTPATH, 'tensordistribution'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    analyze_tensor_sizes()


from __future__ import print_function
from audioop import avg
from msilib.schema import Font
import os
from re import T
import sys
import logging
import time
import copy
import datetime
import itertools

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

import platform
sysstr = platform.system()
if ("Windows" in sysstr):
    matplotlib.use("TkAgg")
    print ("On Windows, matplotlib use TkAgg")
else:
    matplotlib.use("Agg")
    print ("On Windows, matplotlib use Agg")


import numpy as np

from pandas import Series,DataFrame
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))



from utils.plot_util import (
    update_fontsize,
    plot_line_figure,
    draw_ax_legend,
    hex_to_rgb,
    rgb_to_hex,
    rgb_scale
)
from utils.experiment_util import (
    combine_config,
    get_summary_name,
    get_same_alias_metric_things
)

from utils.common import *
from utils.plot_basic import *
from get_results import *

OUTPUTPATH='./'



max_round_dict = {
    cifar10: 200,
    femnist: 200,
    # stackoverflow: 500,
    # reddit: 500,
    # reddit_blog: 500,
}




def plot_trainloss_lines(datas, color_map, markers, linestyles,
                label_list, x_lim, y_lim, x_label, y_label, 
                legend_config, subplots_adjust, file_name, **kwargs):

    fig = plt.figure(figsize=(5, 3.4))
    fontsize = 14
    linewidth = 2.0
    markersize = 6
    ax = fig.gca()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # plt.subplots_adjust(**dict(bottom=0.08, left=0.1, right=0.96, top=0.8))
    # plt.subplots_adjust(**dict(bottom=0.18, left=0.16, right=0.96, top=0.96))
    plt.subplots_adjust(**subplots_adjust)


    for i, data in enumerate(datas):
        plot_kwargs = {}
        if linestyles is not None:
            linestyle=linestyles[i]
            plot_kwargs["linestyle"] = linestyle
        if markers is not None:
            # marker=markers[i % len(label_list)] 
            marker=markers[i]
            plot_kwargs["marker"] = marker
        if color_map is not None:
            color=color_map[i % len(label_list)]
            plot_kwargs["color"] = color
        # ax.plot(data['x'], data['y'], label=label_list[i], linewidth=linewidth, markerfacecolor='none', 
        #         markersize=markersize, **plot_kwargs)
        ax.plot(data['x'], data['y'], label=label_list[i], linewidth=linewidth, markerfacecolor='none', 
                markersize=markersize, markevery=10, **plot_kwargs)

    if x_lim is not None:
        ax.set_xlim(x_lim)
        # plt.xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim)

    # if getattr(kwargs, "log_x", False):
    if kwargs.get("log_x", False):
        ax.set_xscale("log")
        plt.xscale('log')

    ax.grid(linestyle=":")
    ax.legend(fontsize=fontsize, loc="best", ncol=2, labelspacing=0.5, columnspacing=0.5, handletextpad=0.5)
    # formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1, 10000000))
    # # ax.yaxis.set_major_formatter(formatter)
    # ax.xaxis.set_major_formatter(formatter)
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # 强制x轴始终使用科学计数法

    leg = plt.legend()
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()

    for line in leg_lines:
        plt.setp(line, linewidth=2.0)

    update_fontsize(ax, fontsize)
    # plt.savefig(file_name, transparent=True, bbox_inches='tight')
    plt.savefig(file_name)
    # plt.show()



if __name__ == '__main__':

    CIFAR10_RES18 = "CIFAR10_RES18"
    # build_run("hpml-hkbu/DDP-Train/ddlvt4ws", CIFAR10_RES18,
    #         {"": ""}, "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001")
    # build_run("hpml-hkbu/DDP-Train/l9e8lv4q", CIFAR10_RES18,
    # {"": ""}, "sgd-noiFalse-resnet18-SGD-lr0.1")
    # build_run("hpml-hkbu/DDP-Train/bqpotgaf", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001")
    # build_run("hpml-hkbu/DDP-Train/20qh9sdc", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001")
    # build_run("hpml-hkbu/DDP-Train/fvz4m9uu", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01")
    # build_run("hpml-hkbu/DDP-Train/4ulg0zsl", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1")
    # build_run("hpml-hkbu/DDP-Train/id8neble", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0")

    # build_run("hpml-hkbu/DDP-Train/hg1dh6wt", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/3w6hpski", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/u4vwdclx", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/fhrl0agb", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/405yw7ns", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/0yprgovf", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/eyiz046t", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/qg6ztcfw", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/rhzigbpx", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/shjaq05n", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/hgickwsf", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/b178pgbd", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/y0iwspxl", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/cg7l8rwi", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/v16n8u80", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP50")
    # # build_run("hpml-hkbu/DDP-Train/jrvt4f5h", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/3f8q5hmh", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/bjupi4ih", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/fzjevwvk", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/ooewfl95", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/base", CIFAR10_RES18,
    # # {"": ""}, "detect")
    # build_run("hpml-hkbu/DDP-Train/3fjiv1e1", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/q207bqfz", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/iohhnyot", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/5fkl5x9n", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/314fn4s8", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")

    # # build_run("hpml-hkbu/DDP-Train/4-worker", CIFAR10_RES18,
    # # {"": ""}, "Resnet-50")
    # build_run("hpml-hkbu/DDP-Train/lnlehfre", CIFAR10_RES18,
    # {"": ""}, "sgd-noiFalse-resnet50-nw4-SGD-LG20-lr0.1-bs128-")
    # build_run("hpml-hkbu/DDP-Train/9i0s3y9o", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001")
    # build_run("hpml-hkbu/DDP-Train/ogyvemxc", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001")
    # build_run("hpml-hkbu/DDP-Train/ygfr4yr5", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01")
    # build_run("hpml-hkbu/DDP-Train/doi47ga9", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.1")
    # build_run("hpml-hkbu/DDP-Train/7fgwnmzh", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd1.0")

    # build_run("hpml-hkbu/DDP-Train/y69s8mhb", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/s6b6m2jr", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/2uzeb238", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/yd1v5w2t", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/x6o65qxx", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/qnnvyreo", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/14ke3yd7", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/41fbkdtf", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/2jmxhazx", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/tetvndvz", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/la5wej3y", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/83hh1zun", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/qva45urt", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/d8rvwtkk", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP50")
    # # build_run("hpml-hkbu/DDP-Train/vlxjmtrc", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/p8rbp5mb", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/z8rij6nd", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/nif0tprr", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    # # build_run("hpml-hkbu/DDP-Train/z59drayy", CIFAR10_RES18,
    # # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")

    # # build_run("hpml-hkbu/DDP-Train/base", CIFAR10_RES18,
    # # {"": ""}, "detect")
    # build_run("hpml-hkbu/DDP-Train/lhadl6zg", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/v03egkwg", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/gus6lq0p", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/n7wssgrs", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/74g7oapt", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")




    # build_run("hpml-hkbu/DDP-Train/32-worker", CIFAR10_RES18,
    # {"": ""}, "Resnet-50")
    build_run("hpml-hkbu/DDP-Train/1ijmvjh1", CIFAR10_RES18,
    {"": ""}, "sgd-noiFalse-resnet50-nw32-SGD-LG20-lr0.1-bs128-")
    build_run("hpml-hkbu/DDP-Train/kjr0zuig", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001")
    build_run("hpml-hkbu/DDP-Train/945hp3wx", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001")
    build_run("hpml-hkbu/DDP-Train/9v7ch8vu", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01")
    build_run("hpml-hkbu/DDP-Train/qtaycdbw", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1")
    build_run("hpml-hkbu/DDP-Train/6pal1kpl", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0")

    build_run("hpml-hkbu/DDP-Train/alvxajcd", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/3zn8vl8y", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/hromcduy", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP5")
    build_run("hpml-hkbu/DDP-Train/7fb6n0w7", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP5")
    build_run("hpml-hkbu/DDP-Train/okha0hbv", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP5")
    build_run("hpml-hkbu/DDP-Train/t4k8edy2", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10")
    build_run("hpml-hkbu/DDP-Train/0j0fknwt", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP10")
    build_run("hpml-hkbu/DDP-Train/ud20b6zx", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP10")
    build_run("hpml-hkbu/DDP-Train/ugxlaz6z", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP10")
    build_run("hpml-hkbu/DDP-Train/o7tlqijx", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP10")
    build_run("hpml-hkbu/DDP-Train/eerecwm0", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50")
    build_run("hpml-hkbu/DDP-Train/e5r3b4k7", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP50")
    build_run("hpml-hkbu/DDP-Train/l5m9rqyc", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP50")
    build_run("hpml-hkbu/DDP-Train/ye3a6y1j", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP50")
    build_run("hpml-hkbu/DDP-Train/uqn4k4e7", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/tv7ytwda", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/h945heh6", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/ay6yod4r", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/3figsj55", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/8kbv9m9t", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/base", CIFAR10_RES18,
    # {"": ""}, "detect")
    build_run("hpml-hkbu/DDP-Train/pikfppnq", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    build_run("hpml-hkbu/DDP-Train/car3dlwu", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    build_run("hpml-hkbu/DDP-Train/pcjhol47", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    build_run("hpml-hkbu/DDP-Train/cvnjlh3t", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    build_run("hpml-hkbu/DDP-Train/hj1doccl", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES18])
    filter_none(metrics, rounds)

    # metrics, rounds = load_datas(TIME_PER_ITER, ITERS, all_figures[CIFAR10_RES18])
    # filter_none(metrics, rounds)


    datas = []

    i = 1 
    for alias in all_figures[CIFAR10_RES18]:
        filter = rounds[alias] < 80000
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})
        y = np.array(y)
        # y.mean()
        # print(f"{alias}: {y.mean()}")
        print(f"{alias}: {y[-1]*100:.1f}")

    # print(datas)






    # file_name = f"{VAL_ACC}_{simple_name}.pdf"
    # plot_trainloss_lines(datas, color_map=color_map, markers=markers, linestyles=linestyles,
    #             label_list=label_list, x_lim=None, y_lim=None,
    #             x_label="# Epochs", y_label=y_label, 
    #             legend_config=legend_config, subplots_adjust=subplots_adjust, file_name=file_name)


    # layers = ["diver/layer1.1.conv1.weight",
    #         "diver/layer2.1.conv1.weight",
    #         "diver/layer3.1.conv1.weight",
    #         "diver/layer4.1.conv1.weight"
    #         ]

    for l in range(1, 5):
        print(f"diver/layer{l}.1.conv1.weight")


























































































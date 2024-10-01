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

    #Res18
    # 0.1
    build_run("hpml-hkbu/DDP-Train/4okbu1kk", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-0.1burst")

    # 1.0
    build_run("hpml-hkbu/DDP-Train/egqsg78n", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-1.0burst")

    # fix
    # 0.1 
    build_run("hpml-hkbu/DDP-Train/zppeqg6s", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5-fix-0.1burst")
    build_run("hpml-hkbu/DDP-Train/gqlq6yjt", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50-fix-0.1burst")
    build_run("hpml-hkbu/DDP-Train/vt91s6yg", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10-fix-0.1burst")

    # 1.0
    build_run("hpml-hkbu/DDP-Train/q9sa9h4j", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50-fix-1.0burst")
    build_run("hpml-hkbu/DDP-Train/5m3ieefc", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10-fix-1.0burst")
    build_run("hpml-hkbu/DDP-Train/djzds6cs", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5-fix-1.0burst")

    # sync
    # 0.1
    build_run("hpml-hkbu/DDP-Train/j6dudqku", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10-sync-0.1burst")
    # 1.0
    build_run("hpml-hkbu/DDP-Train/wmd0n193", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tburst-resnet18-nw8-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10-sync-1.0burst")





    simple_name = "cifar10_ResNet18"
    markers = [None]*10
    linestyles = [None]*10
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"]*2 + ["--"]*2 + [":"]*2
    # color_map = [
    #     "#990033", "#084081", "#006633", "#3f007d", 
    #     # "#336666", "#663366", "#336699", "#663300", "#F89933",
    # ]
    linestyles = ["-", "--", "-", "--", "-", "--", ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",  
    # ]
    color_map = [
        "#084081", "#084081", "#000000",  
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["WO. Noise", "W. Noise", "W. Noise and PSync"]


    y_label = "Test Accuracy [%]"
    # EPOCHS = "epochs"

    # metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES18])
    # filter_none(metrics, rounds)

    metrics, rounds = load_datas(TIME_PER_ITER, ITERS, all_figures[CIFAR10_RES18])
    filter_none(metrics, rounds)


    datas = []

    i = 1 
    for alias in all_figures[CIFAR10_RES18]:
        filter = rounds[alias] < 80000
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})
        y = np.array(y)
        # y.mean()
        print(f"{alias}: {y.mean()}")

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




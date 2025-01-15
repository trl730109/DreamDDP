from __future__ import print_function
from audioop import avg
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
from matplotlib.ticker import MultipleLocator
import platform

sysstr = platform.system()
if "Windows" in sysstr:
    matplotlib.use("TkAgg")
    print("On Windows, matplotlib use TkAgg")
else:
    matplotlib.use("Agg")
    print("On Windows, matplotlib use Agg")


import numpy as np

from pandas import Series, DataFrame
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
    rgb_scale,
)
from utils.experiment_util import (
    combine_config,
    get_summary_name,
    get_same_alias_metric_things,
)

from utils.common import *
from utils.plot_basic import *
from get_results import *

OUTPUTPATH = "./"

def cal_sgd(total_iterations, train_time, comm_time):
  time_sum = 0
  time_per_iter = train_time + comm_time
  for i in range(total_iterations):
    time_sum += time_per_iter
  return time_sum / total_iterations

def cal_pipe_sgd(total_iterations, train_time, comm_time):
  time_sum = 0
  time_per_iter = train_time + comm_time * 0.6
  for i in range(total_iterations):
    time_sum += time_per_iter
  return time_sum / total_iterations

def cal_localsgd(total_iterations,nsteps_localsgd, train_time, comm_time):
  time_sum = 0
#   train_time = time_dict[dnn][nworkers]['train']
#   comm_time = time_dict[dnn][nworkers]['comm']
  for i in range(total_iterations):
    if(i != 0 and i % nsteps_localsgd == 0):
      time_sum += (train_time + comm_time)
    else:
      time_sum += train_time

  return time_sum / total_iterations


def cal_dreamddp(total_iterations, H, waittime_per_H, train_time):
  time_sum = 0
#   train_time = time_dict[dnn][nworkers]['train']
#   comm_time = time_dict[dnn][nworkers]['comm']
  for i in range(total_iterations):
    if(i != 0 and i % H == 0):
      time_sum += (train_time + waittime_per_H)
    else:
      time_sum += train_time

  return time_sum / total_iterations

def cal_pipe_seq_localsgd(total_iterations, H, wait_list, train_time):
  time_dict = {}
  time_sum = 0
#   train_time = time_dict[dnn][nworkers]['train']
#   comm_time = time_dict[dnn][nworkers]['comm']
  for i in range(total_iterations):
    time_sum += (train_time + wait_list[i%H])

  return time_sum / total_iterations

# [gpt32_bp_sum, gpt32_comm_sum, plsgd_gpt, [wait_gpt,total_iteration_gpt], nsteps_localsgd]
def get_time_list(time_list):
    time_dict = {}
    nsteps_localsgd = time_list[4]
    time_dict["sgd"] = cal_sgd(1000, time_list[0], time_list[1])

    time_dict["pipe_sgd"] = cal_pipe_sgd(1000, time_list[0], time_list[1])

    time_dict["localsgd"] = cal_localsgd(1000, nsteps_localsgd,time_list[0], time_list[1])

    wait_list = time_list[2]
    time_dict["pipe_seq_localsgd"] = cal_pipe_seq_localsgd(1000, time_list[4],wait_list,time_list[0])

    wait_time = time_list[3][0]
    time_dict["dream_ddp"] = cal_dreamddp(1000, time_list[4], wait_time, time_list[0])
    return list(time_dict.values())

def get_simulated_time_list(alias, time_list):
    # sgd, pipe_sgd, localsgd, pipe_seq_localsgd, dream_ddp
    iter_per_epoch_dict = {'resnet18': 13 ,'resnet50':13, 'gpt2':19, 'llama2-124M':22}
    config = get_run(alias).config
    dnn = config['dnn']
    if config['alg'] == "transformer_sgd" or config['alg'] == "sgd":
        time_range = np.arange(config['max_epochs']) * (time_list[0] * iter_per_epoch_dict[dnn])
    elif config['alg'] == "transformer_pipe_sgd" or config['alg'] == "pipe_sgd":
        time_range = np.arange(config['max_epochs']) * (time_list[1] * iter_per_epoch_dict[dnn])
    elif config['alg'] == "transformer_localsgd" or config['alg'] == "localsgd":
        time_range = np.arange(config['max_epochs']) * (time_list[2] * iter_per_epoch_dict[dnn])
    elif config['alg'] == "transformer_pipe_seq_localsgd" or config['alg'] == "pipe_seq_localsgd":
        time_range = np.arange(config['max_epochs']) * (time_list[3] * iter_per_epoch_dict[dnn])
    elif config['alg'] == "transformer_dream_ddp" or config['alg'] == "dream_ddp":
        time_range = np.arange(config['max_epochs']) * (time_list[4] * iter_per_epoch_dict[dnn])
    return time_range

max_round_dict = {
    cifar10: 200,
    femnist: 200,
    # stackoverflow: 500,
    # reddit: 500,
    # reddit_blog: 500,
}

def plot_trainloss_lines_usenix(
    datas,
    color_map,
    markers,
    linestyles,
    linewidth,
    label_list,
    x_lim_max,
    x_lim_min,
    y_lim_max,
    y_lim_min,
    x_label,
    y_label,
    legend_config,
    subplots_adjust,
    file_name,
    markevery=4,
    **kwargs,
):
    # Enhanced figure size for rectangular plots
    fig = plt.figure(figsize=(8, 5))

    # Enhanced font and line configurations
    fontsize = 24
    linewidth = linewidth
    markersize = 10
    ax = fig.gca()

    # Set axis labels with bold font
    ax.set_xlabel(x_label, fontsize=fontsize, weight="bold")
    ax.set_ylabel(y_label, fontsize=fontsize, weight="bold")

    # Adjust subplot margins
    plt.subplots_adjust(**subplots_adjust)

    # Plot each data line
    for i, data in enumerate(datas):
        plot_kwargs = {}
        if linestyles is not None:
            plot_kwargs["linestyle"] = linestyles[i]
        if markers is not None:
            plot_kwargs["marker"] = markers[i]
        if color_map is not None:
            plot_kwargs["color"] = color_map[i % len(label_list)]

        ax.plot(
            data["x"],
            data["y"],
            label=label_list[i],
            linewidth=linewidth,
            markersize=markersize,
            # markerfacecolor="none",
            markevery=markevery,  # Adds markers at intervals 4
            **plot_kwargs, 
        )

    # Set axis limits
    if x_lim_min is not None:
        ax.set_xlim(x_lim_min, x_lim_max)
    if "resnet" in file_name:
        y_lim_min = 0 if y_lim_min is None else y_lim_min
        y_lim_max = 100 if y_lim_max is None else y_lim_max
        ax.set_ylim(y_lim_min, y_lim_max)

    else:
        if y_lim_max is not None:
            ax.set_ylim(0, y_lim_max)



    # Set axis ticks
    ax.tick_params(axis="x", labelsize=fontsize - 4)
    ax.tick_params(axis="y", labelsize=fontsize - 4)
    # ax.yaxis.set_major_locator(MultipleLocator(10))
    # ax.xaxis.set_major_locator(MultipleLocator(300))
    # Configure gridlines
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add legend with customized fonts
    leg = ax.legend(
        fontsize=fontsize - 5,
        loc=legend_config.get("loc", "best"),
        ncol=legend_config.get("ncol", 1),
        frameon=True,
        labelspacing=0.4,
        columnspacing=0.8,
    )

    # Adjust legend line widths
    for line in leg.get_lines():
        plt.setp(line, linewidth=2.0)

    # Enhance axis borders
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("black")

    # Finalize layout and save the plot
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.show()



def plot_trainloss_lines(
    datas,
    color_map,
    markers,
    linestyles,
    label_list,
    x_lim,
    y_lim,
    x_label,
    y_label,
    legend_config,
    subplots_adjust,
    file_name,
    **kwargs,
):

    fig = plt.figure(figsize=(3.5, 2.38))
    fontsize = 10
    linewidth = 1.0
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
            linestyle = linestyles[i]
            plot_kwargs["linestyle"] = linestyle
        if markers is not None:
            # marker=markers[i % len(label_list)]
            marker = markers[i]
            plot_kwargs["marker"] = marker
        if color_map is not None:
            color = color_map[i % len(label_list)]
            plot_kwargs["color"] = color
        # ax.plot(data['x'], data['y'], label=label_list[i], linewidth=linewidth, markerfacecolor='none',
        #         markersize=markersize, **plot_kwargs)
        ax.plot(
            data["x"],
            data["y"],
            label=label_list[i],
            linewidth=linewidth,
            markerfacecolor="none",
            markersize=markersize,
            markevery=10,
            **plot_kwargs,
        )

    if x_lim is not None:
        if isinstance(x_lim, list):
            ax.set_xlim(x_lim[0], x_lim[1])
        else:
            ax.set_xlim(0, x_lim)
        # plt.xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(0, y_lim)

    # if getattr(kwargs, "log_x", False):
    if kwargs.get("log_x", False):
        ax.set_xscale("log")
        plt.xscale("log")

    ax.grid(linestyle=":")
    ax.legend(
        fontsize=fontsize - 3,
        loc="best",
        ncol=2,
        labelspacing=0.3,
        columnspacing=0.3,
        handletextpad=0.5,
    )
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
    plt.tight_layout()
    # plt.savefig(file_name, transparent=True, bbox_inches='tight')
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    # plt.show()


def plot_dual_axis_trainloss(
    datas,
    color_map,
    markers,
    linestyles,
    label_list,
    x_lim,
    y_lim,
    x_label,
    y_label,
    legend_config,
    subplots_adjust,
    file_name,
    secondary_y_data=None,  # Data for the secondary y-axis
    secondary_y_label=None,  # Label for the secondary y-axis
    secondary_y_lim=None,  # Limits for the secondary y-axis
    secondary_label_list=None,  # Label for the secondary y-axis
    secondary_color_map=None,  # Color map for the secondary y-axis
    **kwargs,
):
    fig, ax1 = plt.subplots(figsize=(3.5, 2.38))
    fontsize = 8
    linewidth = 1.0
    markersize = 3

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    plt.subplots_adjust(**subplots_adjust)

    # Plot data on the primary y-axis (left)
    for i, data in enumerate(datas):
        plot_kwargs = {}
        if linestyles is not None:
            plot_kwargs["linestyle"] = "-"
        if markers is not None:
            plot_kwargs["marker"] = markers[i]
        if color_map is not None:
            plot_kwargs["color"] = color_map[i % len(label_list)]

        ax1.plot(
            data["x"],
            data["y"],
            label=label_list[i],
            linewidth=linewidth,
            markerfacecolor="none",
            markersize=markersize,
            markevery=10,
            **plot_kwargs,
        )

    # secondary (right) y-axis
    if secondary_y_data is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(secondary_y_label)
        for i, data in enumerate(secondary_y_data):
            plot_kwargs = {}
            if linestyles is not None:
                plot_kwargs["linestyle"] = "--"
            if markers is not None:
                plot_kwargs["marker"] = markers[i]
            if secondary_color_map is not None:
                plot_kwargs["color"] = secondary_color_map[i % len(label_list)]

            # trasparent to make it easier to see the primary y-axis data

            ax2.plot(
                data["x"],
                data["y"],
                label=secondary_label_list[i],
                linewidth=linewidth,
                markerfacecolor="none",
                markersize=markersize,
                markevery=10,
                alpha=0.5,
                **plot_kwargs,
            )

        if secondary_y_lim is not None:
            if isinstance(secondary_y_lim, list):
                ax2.set_ylim(secondary_y_lim[0], secondary_y_lim[1])
            else:
                ax2.set_ylim(0, secondary_y_lim)

    if x_lim is not None:
        if isinstance(x_lim, list):
            ax1.set_xlim(x_lim[0], x_lim[1])
        else:
            ax1.set_xlim(0, x_lim)

    if y_lim is not None:
        if isinstance(y_lim, list):
            ax1.set_ylim(y_lim[0], y_lim[1])
        else:
            ax1.set_ylim(0, y_lim)

    if kwargs.get("log_x", False):
        ax1.set_xscale("log")

    ax1.grid(linestyle=":")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    if secondary_y_data is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
    else:
        lines, labels = lines1, labels1

    ax1.legend(
        lines,
        labels,
        fontsize=fontsize,
        loc="best",
        ncol=2,
        labelspacing=0.5,
        columnspacing=0.5,
        handletextpad=0.5,
    )

    leg = ax1.get_legend()
    for line in leg.get_lines():
        plt.setp(line, linewidth=2.0)

    update_fontsize(ax1, fontsize)
    if secondary_y_data is not None:
        update_fontsize(ax2, fontsize)

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")


def update_fontsize(ax, fontsize):
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)


# def run_test():
#     CIFAR10_RES = "CIFAR10_RES"
#     build_run(
#         "hpml-hkbu/DDP-Train/g2ef5gy0",
#         CIFAR10_RES,
#         {"": ""},
#         "2Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
#     )

#     simple_name = "cifar10_ResNet18"
#     markers = [None] * 10
#     linestyles = [None] * 10
#     # markers = ['o']*2 + ['v']*2 + ['D']*2
#     linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 2
#     color_map = [
#         "#990033",
#         "#084081",
#         "#006633",
#         "#3f007d",
#         "#336666",
#         "#663366",
#         # "#336699",
#         # "#663300",
#         # "#F89933",
#     ]
#     linestyles = [
#         "-",
#         "--",
#         "-",
#         "--",
#         "-",
#         "--",
#     ]
#     # color_map = [
#     #     "#006633", "#006633", "#990033", "#990033",
#     # ]
#     # color_map = [
#     #     "#084081",
#     #     "#084081",
#     #     "#000000",
#     # ]
#     legend_config = dict(fontsize=10, loc="lower right", ncol=2)
#     subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
#     subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
#     # label_list = ["FedAvg", "FedProx"]
#     # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
#     label_list = ["WO. Noise", "W. Noise", "W. Noise and PSync"]

#     y_label = "Test Accuracy (%)"
#     # EPOCHS = "epochs"

#     metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES])
#     print(f"metrics: {metrics}")
#     print(f"rounds: {rounds}")
#     filter_none(metrics, rounds)
#     datas = []
#     i = 1
#     for alias in all_figures[CIFAR10_RES]:
#         filter = rounds[alias] < 800
#         x = rounds[alias][filter]
#         y = metrics[alias][filter] * 100
#         datas.append({"x": x, "y": y})

#     file_name = f"{VAL_ACC}_{simple_name}.png"
#     plot_trainloss_lines_usenix(
#         datas,
#         color_map=color_map,
#         markers=markers,
#         linestyles=linestyles,
#         label_list=label_list,
#         x_lim=None,
#         y_lim=None,
#         x_label="# Epochs",
#         y_label=y_label,
#         legend_config=legend_config,
#         subplots_adjust=subplots_adjust,
#         file_name=file_name,
    # )

def run_fig10_resnet18_convergence():
    CIFAR10_RES = "CIFAR10_RES"
    build_run(
        "hpml-hkbu/DDP-Train/g2ef5gy0",
        CIFAR10_RES,
        {"": ""},
        "2Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/jexe5tia",
        CIFAR10_RES,
        {"": ""},
        "2Nodes-localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/p9x46qc0",
        CIFAR10_RES,
        {"": ""},
        "2Nodes-pipe_seq_localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/fvb0va8u",
        CIFAR10_RES,
        {"": ""},
        "2Nodes-dream_ddp_5-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    )
    
    simple_name = "cifar10_ResNet18"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#000000',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="lower right", ncol=2)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=35,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig10_resnet18_32workers_convergence():
    CIFAR10_RES_32 = "CIFAR10_RES_32"
    build_run(
        "hpml-hkbu/DDP-Train/pfitqz8u",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/42vyy0d7",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/etltb196",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-pipe_seq_localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/hjg72pls",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-dream_ddp_5-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    
    simple_name = "cifar10_ResNet18_32workers"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#000000',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="lower right", ncol=2)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES_32])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES_32]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=35,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

def run_fig10_resnet50_convergence():
    CIFAR100_RES = "CIFAR100_RES"
    build_run(
        "hpml-hkbu/DDP-Train/g2ef5gy0",
        CIFAR100_RES,
        {"": ""},
        "2Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/jexe5tia",
        CIFAR100_RES,
        {"": ""},
        "2Nodes-localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/p9x46qc0",
        CIFAR100_RES,
        {"": ""},
        "2Nodes-pipe_seq_localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/fvb0va8u",
        CIFAR100_RES,
        {"": ""},
        "2Nodes-dream_ddp_5-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    )

    
    simple_name = "cifar100_ResNet50"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#000000',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="lower right", ncol=2)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR100_RES])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR100_RES]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=35,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig10_resnet50_32workers_convergence():
    CIFAR100_RES_32 = "CIFAR100_RES_32"
    build_run(
        "hpml-hkbu/DDP-Train/uag80cjj",
        CIFAR100_RES_32,
        {"": ""},
        "8Nodes-sgd-resnet50-cifar100-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/v6unhxo7",
        CIFAR100_RES_32,
        {"": ""},
        "Nodes-localsgd-resnet50-cifar100-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/45kq9gy1",
        CIFAR100_RES_32,
        {"": ""},
        "8Nodes-pipe_seq_localsgd-resnet50-cifar100-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/2woabsoj",
        CIFAR100_RES_32,
        {"": ""},
        "8Nodes-dream_ddp_5-resnet50-cifar100-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )

    
    simple_name = "cifar100_ResNet50_32workers"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#000000',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="lower right", ncol=2)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR100_RES_32])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR100_RES_32]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=35,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig13_resnet18_dreamddp_32():
    CIFAR10_RES_32 = "CIFAR10_RES_32"
    build_run(
        "hpml-hkbu/DDP-Train/pfitqz8u",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/42vyy0d7",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    
    build_run(
        "hpml-hkbu/DDP-Train/hjg72pls",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-dream_ddp_5-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/m37rqtbk",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-dream_ddp_10-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/kiyatwgc",
        CIFAR10_RES_32,
        {"": ""},
        "8Nodes-dream_ddp_15-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    
    simple_name = "dreamddp_ResNet18_32workers"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o', 'v', 'D', '^', 's']
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
        "#000000", "#084081", "#800000","#CC0000",
        "#FF6666",  "#336699", "#663300", "#F89933",
    ]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="lower right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD H=10", "DreamDDP H=5", "DreamDDP H=10", "DreamDDP H=20"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES_32])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES_32]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=35,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig13_resnet50_32workers_dreamddp():
    CIFAR100_RES_32 = "CIFAR100_RES_32"
    build_run(
        "hpml-hkbu/DDP-Train/uag80cjj",
        CIFAR100_RES_32,
        {"": ""},
        "8Nodes-sgd-resnet50-cifar100-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/v6unhxo7",
        CIFAR100_RES_32,
        {"": ""},
        "Nodes-localsgd-resnet50-cifar100-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/2woabsoj",
        CIFAR100_RES_32,
        {"": ""},
        "8Nodes-dream_ddp_5-resnet50-cifar100-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/ikmft62n",
        CIFAR100_RES_32,
        {"": ""},
        "8Nodes-dream_ddp_10-resnet50-cifar100-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/mf6w9p7g",
        CIFAR100_RES_32,
        {"": ""},
        "8Nodes-dream_ddp_20-resnet50-cifar100-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )

    
    simple_name = "dreamddp_ResNet50_32workers"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o', 'v', 'D', '^', 's']
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
        "#000000", "#084081", "#800000","#CC0000",
        "#FF6666",  "#336699", "#663300", "#F89933",
    ]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]

    legend_config = dict(fontsize=14, loc="upper left", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD H=10", "DreamDDP H=5", "DreamDDP H=10", "DreamDDP H=20"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR100_RES_32])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR100_RES_32]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=35,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

def run_fig10_gpt2_convergence():
    GPT2_WIKITEXT = "GPT2_WIKITEXT"
    build_run(
        "hpml-hkbu/DDP-Train/i11cpj15",
        GPT2_WIKITEXT,
        {"": ""},
        "gpt2-transformer_sgd-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes1-nworkers4",
    )
    build_run(
        "hpml-hkbu/DDP-Train/m5yq8i6e",
        GPT2_WIKITEXT,
        {"": ""},
        "gpt2-transformer_localsgd-gpt2-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes1-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/0p7z2o8q",
        GPT2_WIKITEXT,
        {"": ""},
        "gpt2-transformer_pipe_seq_localsgd-gpt2-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes1-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/9rv46zg3",
        GPT2_WIKITEXT,
        {"": ""},
        "gpt2-enlarge-transformer_dream_ddp-5-gpt2-true-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes1-nworkers8",
    )

    
    simple_name = "gpt2_wikitext2"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#000000',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="upper right", ncol=2)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Loss"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TEST_LOSS, EPOCHS, all_figures[GPT2_WIKITEXT])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[GPT2_WIKITEXT]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 1 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})
    print(f"datas: {datas}")
    file_name = f"{TEST_LOSS}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        markevery=2,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

def run_fig10_gpt2_32workers_convergence():
    GPT2_32_WIKITEXT = "GPT2_32_WIKITEXT"
    build_run(
        "hpml-hkbu/DDP-Train/9zuszpd3",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-load-pretrain-transformer_sgd-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/y0hpv6v3",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-transformer_localsgd-gpt2-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/jsrslqxg",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-load-pretrain-transformer_pipe_seq_localsgd-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/elqrb96a",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-transformer_dream_ddp_5_enlarge-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )

    
    simple_name = "gpt2_32_wikitext2"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#000000',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="upper right", ncol=2)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Loss"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TEST_LOSS, EPOCHS, all_figures[GPT2_32_WIKITEXT])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[GPT2_32_WIKITEXT]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 1 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})
    print(f"datas: {datas}")
    file_name = f"{TEST_LOSS}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        markevery=2,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig13_gpt2_32workers_dreamddp():
    GPT2_32_WIKITEXT = "GPT2_32_WIKITEXT"
    build_run(
        "hpml-hkbu/DDP-Train/9zuszpd3",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-load-pretrain-transformer_sgd-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/y0hpv6v3",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-transformer_localsgd-gpt2-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/elqrb96a",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-transformer_dream_ddp_5_enlarge-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/2cv2ua5i",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-transformer_dream_ddp_10_enlarge-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/27rie8h7",
        GPT2_32_WIKITEXT,
        {"": ""},
        "gpt-transformer_dream_ddp_20_enlarge-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )

    
    simple_name = "dreamddp_gpt2_32"
    markers = ['o', 'v', 'D', '^', 's']
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    color_map = [
        "#000000", "#084081", "#800000","#CC0000",
        "#FF6666",  "#336699", "#663300", "#F89933",
    ]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="upper right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD H=10", "DreamDDP H=5", "DreamDDP H=10", "DreamDDP H=20"]

    y_label = "Test Loss"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TEST_LOSS, EPOCHS, all_figures[GPT2_32_WIKITEXT])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[GPT2_32_WIKITEXT]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 1 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})
    print(f"datas: {datas}")
    file_name = f"{TEST_LOSS}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        markevery=2,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

    
def run_fig10_llama2_convergence(): 
    LLAMA2_WIKITEXT = "LLAMA2_WIKITEXT"
    build_run(
        "hpml-hkbu/DDP-Train/2pbr4x3d",
        LLAMA2_WIKITEXT,
        {"": ""},
        "llama2-124M-transformer_pipe_sgd-llama2-124M-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes1-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/h0i4tx6n",
        LLAMA2_WIKITEXT,
        {"": ""},
        "llama2-124M-transformer_localsgd-llama2-124M-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes1-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/0m9p3cvw",
        LLAMA2_WIKITEXT,
        {"": ""},
        "llama2-124M-transformer_pipe_seq_localsgd-llama2-124M-wikitext2-20-1G-lr0.0001-lr_decayfixed-nodes1-nworkers8",
    )
    build_run(
        "hpml-hkbu/DDP-Train/utwjiob8",
        LLAMA2_WIKITEXT,
        {"": ""},
        "llama2-124M-enlarge-transformer_dream_ddp-20-llama2-124M-true-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes1-nworkers8",
    )

    
    simple_name = "llama2_wikitext2"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#000000',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="upper right", ncol=2)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Loss"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TEST_LOSS, EPOCHS, all_figures[LLAMA2_WIKITEXT])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[LLAMA2_WIKITEXT]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 1 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})
    print(f"datas: {datas}")
    file_name = f"{TEST_LOSS}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        markevery=2,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

def run_fig10_llama2_32workers_convergence(): 
    LLAMA2_32_WIKITEXT = "LLAMA2_32_WIKITEXT"
    build_run(
        "hpml-hkbu/DDP-Train/2b721qum",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-transformer_sgd-llama2-124M-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/628hygyq",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-transformer_localsgd-llama2-124M-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/aqvljtrh",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-transformer_pipe_seq_localsgd-llama2-124M-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/i22v47je",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-enlarge-transformer_dream_ddp-llama2-124M-true-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )

    
    simple_name = "llama2_32_wikitext2"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#000000',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="upper right", ncol=2)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Loss"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TEST_LOSS, EPOCHS, all_figures[LLAMA2_32_WIKITEXT])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[LLAMA2_32_WIKITEXT]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 1 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})
    print(f"datas: {datas}")
    file_name = f"{TEST_LOSS}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        markevery=2,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig13_llama2_32workers_dreamddp(): 
    LLAMA2_32_WIKITEXT = "LLAMA2_32_WIKITEXT"
    build_run(
        "hpml-hkbu/DDP-Train/2b721qum",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-transformer_sgd-llama2-124M-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/628hygyq",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-transformer_localsgd-llama2-124M-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )

    build_run(
        "hpml-hkbu/DDP-Train/i22v47je",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-enlarge-transformer_dream_ddp-llama2-124M-true-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/10kq69i9",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-enlarge-transformer_dream_ddp_10-llama2-124M-true-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/mgm4bm1y",
        LLAMA2_32_WIKITEXT,
        {"": ""},
        "llama2-124M-enlarge-transformer_dream_ddp_20-llama2-124M-true-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32",
    )

    
    simple_name = "dreamddp_llama2_32"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o', 'v', 'D', '^', 's']
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
        "#000000", "#084081", "#800000","#CC0000",
        "#FF6666",  "#336699", "#663300", "#F89933",
    ]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=14, loc="upper right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "FLSGD H=10", "DreamDDP H=5", "DreamDDP H=10", "DreamDDP H=20"]

    y_label = "Test Loss"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TEST_LOSS, EPOCHS, all_figures[LLAMA2_32_WIKITEXT])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[LLAMA2_32_WIKITEXT]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 1 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})
    print(f"datas: {datas}")
    file_name = f"{TEST_LOSS}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        markevery=2,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
    
def run_fig5_resnet18_convergence():
    CIFAR10_RES = "CIFAR10_RES"
    build_run(
        "hpml-hkbu/DDP-Train/pfitqz8u",
        CIFAR10_RES,
        {"": ""},
        "8Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/etltb196",
        CIFAR10_RES,
        {"": ""},
        "8Nodes-pipe_seq_localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/42vyy0d7",
        CIFAR10_RES,
        {"": ""},
        "8Nodes-localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    # build_run(
    #     "hpml-hkbu/DDP-Train/fvb0va8u",
    #     CIFAR10_RES,
    #     {"": ""},
    #     "2Nodes-dream_ddp_5-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    # )
    
    simple_name = "resnet18_convergence"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#FFA500',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=12, loc="lower right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "PLSGD ENP H=10", "PLSGD H=10"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=25,
        y_lim_max=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig5_resnet18_diversity():
    CIFAR10_RES = "CIFAR10_RES"
    build_run(
        "hpml-hkbu/DDP-Train/v649ztww",
        CIFAR10_RES,
        {"": ""},
        "Divergence-8Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/tq73p6jc",
        CIFAR10_RES,
        {"": ""},
        "diversity_check_8Nodes-pipe_seq_localsgd-resnet18-cifar10-10-10G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    build_run(
        "hpml-hkbu/DDP-Train/lgkruq9r",
        CIFAR10_RES,
        {"": ""},
        "diversity_check_8Nodes-localsgd-resnet18-cifar10-10-10G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    # build_run(
    #     "hpml-hkbu/DDP-Train/fvb0va8u",
    #     CIFAR10_RES,
    #     {"": ""},
    #     "2Nodes-dream_ddp_5-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes2-nworkers8",
    # )
    
    simple_name = "resnet18_diversity"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    # markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    markers = [None] * 5
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
    '#FFA500',  # Black
    '#800000',  # Darker Red
    '#004d00',  # Darker Green
    '#FFA500',  # Brighter Orange
]

    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=12, loc="upper right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "PLSGD ENP H=10", "PLSGD H=10"]

    y_label = "Total Diversity"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TOTAL_DIVER, EPOCHS, all_figures[CIFAR10_RES])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=-0.0001,
        y_lim_max=0.0020,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )


def run_fig12_llama2():
    llama_time = [0.5524348390513454, 4.684302508831024, [0.4543073200000001, 0.02239168999999999, 0.06523888999999997, 0.001979409999999973, 0.06518594999999999, 0.12209888999999999, 0.28413884000000006, 0.33770208, 0.46829199, 0.52348524], [1.7342929000000005, 10], 10]
    time_list = get_time_list(llama_time)
    
    LLAMA2_WIKITEXT2 = "LLAMA2_WIKITEXT2"
    build_run("hpml-hkbu/DDP-Train/2b721qum", LLAMA2_WIKITEXT2,
                {"": ""}, "llama2-124M-transformer_sgd-llama2-124M-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    # pipe_sgd
    build_run("hpml-hkbu/DDP-Train/5bzwsj71", LLAMA2_WIKITEXT2,
            {"": ""}, "llama2-124M-transformer_pipe_sgd-llama2-124M-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    # localsgd
    build_run("hpml-hkbu/DDP-Train/qcnjuhvm", LLAMA2_WIKITEXT2,
            {"": ""}, "llama2-124M-transformer_localsgd-llama2-124M-wikitext2-5-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    # pipe_seq_localsgd
    build_run("hpml-hkbu/DDP-Train/5shd2sya", LLAMA2_WIKITEXT2,
            {"": ""}, "llama2-124M-transformer_pipe_seq_localsgd-llama2-124M-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    # dream_ddp
    build_run("hpml-hkbu/DDP-Train/i22v47je", LLAMA2_WIKITEXT2,
            {"": ""}, "llama2-124M-enlarge-transformer_dream_ddp-llama2-124M-true-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    
    simple_name = "llama2_wallclock_time"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    # markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    markers = ['o', 'v', 'D', '^', 's']
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
        "#990033", "#084081", "#006633", "#3f007d",
        "#F89933",  "#336699", "#663300", "#F89933",
    ]
    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]

    legend_config = dict(fontsize=14, loc="upper right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "ASC-WFBP", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Loss"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TEST_LOSS, EPOCHS, all_figures[LLAMA2_WIKITEXT2])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[LLAMA2_WIKITEXT2]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 1 == 0)
        x = get_simulated_time_list(alias, time_list)
        x = x[filter]
        print(f"x: {x}")
        y = metrics[alias][filter]
        if (get_run(alias).config['alg'] == 'transformer_sgd'):
            sgd_alias = alias
        if (get_run(alias).config['alg'] == 'transformer_pipe_sgd'):
            y = metrics[sgd_alias][filter]
        datas.append({"x": x, "y": y})

    file_name = f"{TEST_LOSS}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=None,
        x_label="Wall-Clock Time (s)",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig12_gpt2():
    gpt2_time = [0.6478173112216061, 4.569759879793439, [0.5654301299999996, 0, 0, 0, 0, 0.014906349999999995, 0.07321345000000001, 0.08182916000000001, 0.31749743, 1.1587493700000002], [1.8597458099999997, 10], 10]
    time_list = get_time_list(gpt2_time)
    
    GPT2_WIKITEXT2 = "GPT2_WIKITEXT2"
    build_run("hpml-hkbu/DDP-Train/9zuszpd3", GPT2_WIKITEXT2,
                {"": ""}, "gpt-load-pretrain-transformer_sgd-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    # pipe_sgd
    build_run("hpml-hkbu/DDP-Train/dqwzgni7", GPT2_WIKITEXT2,
            {"": ""}, "gpt-load-pretrain-transformer_pipe_sgd-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    # localsgd
    build_run("hpml-hkbu/DDP-Train/y0hpv6v3", GPT2_WIKITEXT2,
            {"": ""}, "gpt-transformer_localsgd-gpt2-wikitext2-10-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    # pipe_seq_localsgd
    build_run("hpml-hkbu/DDP-Train/jsrslqxg", GPT2_WIKITEXT2,
            {"": ""}, "gpt-load-pretrain-transformer_pipe_seq_localsgd-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    # dream_ddp
    build_run("hpml-hkbu/DDP-Train/elqrb96a", GPT2_WIKITEXT2,
            {"": ""}, "gpt-transformer_dream_ddp_5_enlarge-gpt2-wikitext2-nstepsupdate1-1G-lr0.0001-lr_decayfixed-nodes4-nworkers32")
    
    simple_name = "gpt2_wallclock_time"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    # markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    markers = ['o', 'v', 'D', '^', 's']
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
        "#990033", "#084081", "#006633", "#3f007d",
        "#F89933",  "#336699", "#663300", "#F89933",
    ]
    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]

    legend_config = dict(fontsize=14, loc="upper right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "ASC-WFBP", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Loss"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TEST_LOSS, EPOCHS, all_figures[GPT2_WIKITEXT2])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[GPT2_WIKITEXT2]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 1 == 0)
        x = get_simulated_time_list(alias, time_list)
        x = x[filter]
        print(f"x: {x}")
        y = metrics[alias][filter]
        if (get_run(alias).config['alg'] == 'transformer_sgd'):
            sgd_alias = alias
        if (get_run(alias).config['alg'] == 'transformer_pipe_sgd'):
            y = metrics[sgd_alias][filter]
        datas.append({"x": x, "y": y})

    file_name = f"{TEST_LOSS}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=None,
        x_label="Wall-Clock Time (s)",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

def run_fig12_resnet18():
    resnet18_time = [0.07756578005277194, 0.8036470413208008, [0.09549521, 0.26543907, 0.07107025, 0.007627540000000002, 0.02234838000000001, 0, 0, 0, 0.0034697099999999995, 0.00721359], [0.34234606, 10], 10]
    time_list = get_time_list(resnet18_time)
    
    RESNET18_CIFAR10 = "RESNET18_CIFAR10"
    build_run("hpml-hkbu/DDP-Train/pfitqz8u", RESNET18_CIFAR10,
                {"": ""}, "8Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    # pipe_sgd
    build_run("hpml-hkbu/DDP-Train/deyhthqc", RESNET18_CIFAR10,
            {"": ""}, "Divergence-8Nodes-pipe_sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    # localsgd
    build_run("hpml-hkbu/DDP-Train/42vyy0d7", RESNET18_CIFAR10,
            {"": ""}, "8Nodes-localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    # pipe_seq_localsgd
    build_run("hpml-hkbu/DDP-Train/etltb196", RESNET18_CIFAR10,
            {"": ""}, "8Nodes-pipe_seq_localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    # dream_ddp
    build_run("hpml-hkbu/DDP-Train/m37rqtbk", RESNET18_CIFAR10,
            {"": ""}, "8Nodes-dream_ddp_10-resnet18-cifar10-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    
    simple_name = "resnet18_wallclock_time"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    # markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    markers = ['o', 'v', 'D', '^', 's']
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
        "#990033", "#084081", "#006633", "#3f007d",
        "#F89933",  "#336699", "#663300", "#F89933",
    ]
    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]

    legend_config = dict(fontsize=14, loc="lower right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "ASC-WFBP", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[RESNET18_CIFAR10])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[RESNET18_CIFAR10]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 7 == 0)
        x = get_simulated_time_list(alias, time_list)
        x = x[filter]
        print(f"x: {x}")
        y = metrics[alias][filter] * 100
        if (get_run(alias).config['alg'] == 'transformer_sgd'):
            sgd_alias = alias
        if (get_run(alias).config['alg'] == 'transformer_pipe_sgd'):
            y = metrics[sgd_alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=20,
        y_lim_max=None,
        x_label="Wall-Clock Time (s)",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
def run_fig12_resnet50():
    resnet50_time = [0.13466653457054722, 1.6430895328521729, [0.24272801, 0.49352891000000004, 0.04644045999999999, 0.031222290000000007, 0.09371819, 0.019986030000000005, 0, 0, 0, 0.01325653], [0.6161704699999999, 10], 10]
    time_list = get_time_list(resnet50_time)
    
    RESNET50_CIFAR10 = "RESNET50_CIFAR10"
    build_run("hpml-hkbu/DDP-Train/uag80cjj", RESNET50_CIFAR10,
                {"": ""}, "8Nodes-sgd-resnet50-cifar100-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    # pipe_sgd
    build_run("hpml-hkbu/DDP-Train/nk23tqbq", RESNET50_CIFAR10,
            {"": ""}, "8Nodes-pipe_sgd-resnet50-cifar100-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    # localsgd
    build_run("hpml-hkbu/DDP-Train/v6unhxo7", RESNET50_CIFAR10,
            {"": ""}, "Nodes-localsgd-resnet50-cifar100-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    # pipe_seq_localsgd
    build_run("hpml-hkbu/DDP-Train/45kq9gy1", RESNET50_CIFAR10,
            {"": ""}, "8Nodes-pipe_seq_localsgd-resnet50-cifar100-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    # dream_ddp
    build_run("hpml-hkbu/DDP-Train/2woabsoj", RESNET50_CIFAR10,
            {"": ""}, "8Nodes-dream_ddp_5-resnet50-cifar100-20-1G-lr0.1-lr_decayexp-nodes8-nworkers32")
    
    simple_name = "resnet50_wallclock_time"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = [None] + ['o']*3 + ['v']*3 + ['D']*2
    # markers = ['o']*1 + ["v"]*1 + ["D"]*1 + ['^']*1
    markers = ['o', 'v', 'D', '^', 's']
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-"] * 1 + [":"] * 3 + ["--"] * 3 
#     color_map = [
#     "#000000",  # Black (Baseline)
#     "#003366",  # Dark Blue (Deep Navy Blue)
#     "#0073E6",  # Mid Blue (Bright Royal Blue)
#     "#66B2FF",  # Light Blue (Sky Blue)
#     "#800000",  # Dark Red (Deep Maroon)
#     "#CC0000",  # Mid Red (Bright Crimson)
#     "#FF6666",  # Light Red (Soft Coral Red)
# ]
    color_map = [
        "#990033", "#084081", "#006633", "#3f007d",
        "#F89933",  "#336699", "#663300", "#F89933",
    ]
    linestyles = [
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]

    legend_config = dict(fontsize=14, loc="lower right", ncol=1)
    # subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)
    subplots_adjust = dict(bottom=0.15, left=0.15, right=0.95, top=0.95)
    
    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["S-SGD", "ASC-WFBP", "FLSGD", "PLSGD", "DreamDDP"]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[RESNET50_CIFAR10])
    # print(f"metrics: {metrics}")
    # print(f"rounds: {rounds}")
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[RESNET50_CIFAR10]:
        # filter = rounds[alias] < 800
        filter = (rounds[alias] < 800) & (np.arange(len(rounds[alias])) % 7 == 0)
        x = get_simulated_time_list(alias, time_list)
        x = x[filter]
        print(f"x: {x}")
        y = metrics[alias][filter] * 100
        if (get_run(alias).config['alg'] == 'transformer_sgd'):
            sgd_alias = alias
        if (get_run(alias).config['alg'] == 'transformer_pipe_sgd'):
            y = metrics[sgd_alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines_usenix(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        linewidth=2.0,
        label_list=label_list,
        x_lim_min=None,
        x_lim_max=None,
        y_lim_min=None,
        y_lim_max=73,
        x_label="Wall-Clock Time (s)",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

if __name__ == "__main__":
    # run_fig2_b_divergence()
    # run_fig2_a_convergence()
    # run_fig4_a_convergence()
    # run_fig4_b_divergence()
    # run_fig5_a_divergence()
    # run_fig5_b_dual_y()
    # run_fig10_gpt2_32workers_convergence()
    # run_fig5_resnet18_convergence()
    run_fig13_gpt2_32workers_dreamddp()
    # WORKERS = [4, 32]
    # MODELS = ["resnet18", "resnet50", "gpt2"]
    # # , "llama2"]

    # for worker in WORKERS:
    #     for model in MODELS:
    #         # try:
    #         run_convergence_vs_worker_modeltype(worker, model)
    #         # except Exception as e:
    #         #     breakpoint()

    # # for worker in WORKERS:
    # #     for model in MODELS:
    # #         run_convergence_vs_worker_modeltype_fixed_noise(worker, model)

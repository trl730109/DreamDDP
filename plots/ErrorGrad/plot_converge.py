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


max_round_dict = {
    cifar10: 200,
    femnist: 200,
    # stackoverflow: 500,
    # reddit: 500,
    # reddit_blog: 500,
}


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


def run_test():
    CIFAR10_RES = "CIFAR10_RES"
    build_run(
        "hpml-hkbu/DDP-Train/ddlvt4ws",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001",
    )

    simple_name = "cifar10_ResNet18"
    markers = [None] * 10
    linestyles = [None] * 10
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 2
    color_map = [
        "#990033",
        "#084081",
        "#006633",
        "#3f007d",
        "#336666",
        "#663366",
        # "#336699",
        # "#663300",
        # "#F89933",
    ]
    linestyles = [
        "-",
        "--",
        "-",
        "--",
        "-",
        "--",
    ]
    # color_map = [
    #     "#006633", "#006633", "#990033", "#990033",
    # ]
    # color_map = [
    #     "#084081",
    #     "#084081",
    #     "#000000",
    # ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    # label_list = ["FedAvg", "FedProx"]
    # label_list = [r"$E=10 \ a=10$", r"$E=1 \ a=10$", r"$E=10 \ a=0.1$", r"$E=1 \ a=0.1$"]
    label_list = ["WO. Noise", "W. Noise", "W. Noise and PSync"]

    y_label = "Test Accuracy (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 800
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=None,
        y_lim=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

    for l in range(1, 5):
        print(f"diver/layer{l}.1.conv1.weight")


def run_fig2_a_convergence():
    # SGD + Noise x 5 (6 lines)
    # convergence is denoted by test acc

    CIFAR10_RES = "fig2_a_convergence"

    # SGD
    build_run(
        "hpml-hkbu/DDP-Train/l9e8lv4q",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiFalse-resnet18-SGD-lr0.1",
    )

    # Noise 0.0001
    build_run(
        "hpml-hkbu/DDP-Train/bqpotgaf",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001",
    )

    # Noise 0.001
    build_run(
        "hpml-hkbu/DDP-Train/20qh9sdc",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001",
    )

    # Noise 0.01
    build_run(
        "hpml-hkbu/DDP-Train/fvz4m9uu",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01",
    )

    # Noise 0.1
    build_run(
        "hpml-hkbu/DDP-Train/4ulg0zsl",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1",
    )

    # # Noise 1
    # build_run(
    #     "hpml-hkbu/DDP-Train/id8neble",
    #     CIFAR10_RES,
    #     {"": ""},
    #     "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0",
    # )

    simple_name = "cifar10_ResNet18_4workers_fig2_a_convergence"
    markers = [None] * 6
    linestyles = [None] * 6
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 2
    linestyles = [
        "-",
        "--",
        "-",
        "--",
        "-",
        "--",
    ]
    color_map = [
        "#990033",
        "#084081",
        "#006633",
        "#3f007d",
        "#F89933",
        "#663366",
        # "#336699",
        # "#663300",
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "SGD",
        "Noise 0.0001",
        "Noise 0.001",
        "Noise 0.01",
        "Noise 0.1",
        "Noise 1",
    ]

    y_label = "Test Accuracy (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 110
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=None,
        y_lim=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

    for l in range(1, 5):
        print(f"diver/layer{l}.1.conv1.weight")


def run_fig2_b_divergence():
    # SGD + Noise x 5 (6 lines)
    # divergence is denoted by weight norm

    CIFAR10_RES = "fig2_b_divergence"

    # SGD
    build_run(
        "hpml-hkbu/DDP-Train/l9e8lv4q",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiFalse-resnet18-SGD-lr0.1",
    )

    # Noise 0.0001
    build_run(
        "hpml-hkbu/DDP-Train/bqpotgaf",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001",
    )

    # Noise 0.001
    build_run(
        "hpml-hkbu/DDP-Train/20qh9sdc",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001",
    )

    # Noise 0.01
    build_run(
        "hpml-hkbu/DDP-Train/fvz4m9uu",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01",
    )

    # Noise 0.1
    build_run(
        "hpml-hkbu/DDP-Train/4ulg0zsl",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1",
    )

    # Noise 1
    # build_run(
    #     "hpml-hkbu/DDP-Train/id8neble",
    #     CIFAR10_RES,
    #     {"": ""},
    #     "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0",
    # )

    simple_name = "cifar10_ResNet18_4workers_fig2_b_divergence"
    markers = [None] * 6
    linestyles = [None] * 6
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 2
    linestyles = [
        "-",
        "--",
        "-",
        "--",
        "-",
        "--",
    ]
    color_map = [
        "#990033",
        "#084081",
        "#006633",
        "#3f007d",
        "#F89933",
        "#663366",
        # "#336699",
        # "#663300",
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "SGD",
        "Noise 0.0001",
        "Noise 0.001",
        "Noise 0.01",
        "Noise 0.1",
        "Noise 1",
    ]

    y_label = "TOTAL DIVERGENCE"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TOTAL_DIVER, ITERS, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 100000
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=None,
        y_lim=6,
        x_label="# ITERATIONS",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )

    for l in range(1, 5):
        print(f"diver/layer{l}.1.conv1.weight")


def run_fig4_a_convergence():
    CIFAR10_RES = "fig4_a_convergence"

    # SGD
    build_run(
        "hpml-hkbu/DDP-Train/l9e8lv4q",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiFalse-resnet18-SGD-lr0.1",
    )

    # Noise 0.0001
    build_run(
        "hpml-hkbu/DDP-Train/bqpotgaf",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001",
    )

    # Noise 0.01
    build_run(
        "hpml-hkbu/DDP-Train/fvz4m9uu",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01",
    )

    # Noise 0.0001 + SyncP5
    build_run(
        "hpml-hkbu/DDP-Train/hg1dh6wt",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5",
    )

    # Noise 0.01 + SyncP5
    build_run(
        "hpml-hkbu/DDP-Train/u4vwdclx",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP5",
    )

    # Noise 0.0001 + SyncP50
    build_run(
        "hpml-hkbu/DDP-Train/hgickwsf",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50",
    )

    # Noise 0.01 + SyncP50
    build_run(
        "hpml-hkbu/DDP-Train/y0iwspxl",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP50",
    )

    simple_name = "cifar10_ResNet18_4workers_fig4_a_convergence"
    markers = [None] * 7
    linestyles = [None] * 7
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    linestyles = ["-", "--", "-", "--", "-", "--", "-"]
    color_map = [
        "#990033",
        "#084081",
        "#006633",
        "#3f007d",
        "#F89933",
        "#663366",
        "#663300",
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "SGD",
        "Noise 0.0001",
        "Noise 0.01",
        "Noise 0.0001(H5)",
        "Noise 0.01(H5)",
        "Noise 0.0001(H50)",
        "Noise 0.01(H50)",
    ]

    y_label = "Test Accuracy (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)

    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 10000
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    # breakpoint()
    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=110,
        y_lim=None,
        x_label="# EPOCHS",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )


def run_fig4_b_divergence():
    CIFAR10_RES = "fig4_b_divergence"

    # SGD
    build_run(
        "hpml-hkbu/DDP-Train/l9e8lv4q",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiFalse-resnet18-SGD-lr0.1",
    )

    # Noise 0.0001
    build_run(
        "hpml-hkbu/DDP-Train/bqpotgaf",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001",
    )

    # Noise 0.01
    build_run(
        "hpml-hkbu/DDP-Train/fvz4m9uu",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01",
    )

    # Noise 0.0001 + SyncP5
    build_run(
        "hpml-hkbu/DDP-Train/hg1dh6wt",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5",
    )

    # Noise 0.01 + SyncP5
    build_run(
        "hpml-hkbu/DDP-Train/u4vwdclx",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP5",
    )

    # Noise 0.0001 + SyncP50
    build_run(
        "hpml-hkbu/DDP-Train/hgickwsf",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50",
    )

    # Noise 0.01 + SyncP50
    build_run(
        "hpml-hkbu/DDP-Train/y0iwspxl",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP50",
    )

    simple_name = "cifar10_ResNet18_4workers_fig4_b_divergence"
    markers = [None] * 7
    linestyles = [None] * 7
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    linestyles = ["-", "--", "-", "--", "-", "--", "-"]
    color_map = [
        "#990033",
        "#084081",
        "#006633",
        "#3f007d",
        "#F89933",
        "#663366",
        "#663300",
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "SGD",
        "Noise 0.0001",
        "Noise 0.01",
        "Noise 0.0001(H5)",
        "Noise 0.01(H5)",
        "Noise 0.0001(H50)",
        "Noise 0.01(H50)",
    ]

    y_label = "TOTAL DIVERGENCE"

    metrics, rounds = load_datas(TOTAL_DIVER, ITERS, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 10000
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=1000,
        y_lim=0.06,
        x_label="# ITERATIONS",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )


def run_fig5_a_divergence():
    CIFAR10_RES = "fig5_a_divergence"

    # P10 Noise 0.0001 0.001 0.01
    build_run(
        "hpml-hkbu/DDP-Train/0yprgovf",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10",
    )
    build_run(
        "hpml-hkbu/DDP-Train/eyiz046t",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP10",
    )
    build_run(
        "hpml-hkbu/DDP-Train/qg6ztcfw",
        CIFAR10_RES,
        {"": ""},
        "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP10",
    )

    # P100 Noise 0.0001 0.001 0.01
    build_run(
        "hpml-hkbu/DDP-Train/3fjiv1e1",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100",
    )
    build_run(
        "hpml-hkbu/DDP-Train/q207bqfz",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100",
    )
    build_run(
        "hpml-hkbu/DDP-Train/iohhnyot",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100",
    )

    simple_name = "cifar10_ResNet18_4workers_fig5_a_divergence"
    markers = [None] * 6
    linestyles = [None] * 6
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 2
    linestyles = ["-", "--", "-", "--", "-", "--"]
    color_map = [
        "#990033",
        "#084081",
        "#006633",
        "#3f007d",
        "#F89933",
        "#663366",
        # "#663300",
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "Noise 0.0001 (H10)",
        "Noise 0.001 (H10)",
        "Noise 0.01 (H10)",
        "Detect Base Noise 0.001",
        "Detect Base Noise 0.01",
        "Detect Base Noise 0.1",
    ]

    y_label = "TOTAL DIVERGENCE"

    metrics, rounds = load_datas(TOTAL_DIVER, ITERS, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)
    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 10000
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"
    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=[600, 700],
        y_lim=0.02,
        x_label="# ITERATIONS",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )


def run_fig5_b_dual_y():
    CIFAR10_RES = "fig5_b_dual_y"

    # P10 Noise 0.0001 0.001 0.01
    # build_run(
    #     "hpml-hkbu/DDP-Train/0yprgovf",
    #     CIFAR10_RES,
    #     {"": ""},
    #     "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10",
    # )
    # build_run(
    #     "hpml-hkbu/DDP-Train/eyiz046t",
    #     CIFAR10_RES,
    #     {"": ""},
    #     "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP10",
    # )
    # build_run(
    #     "hpml-hkbu/DDP-Train/qg6ztcfw",
    #     CIFAR10_RES,
    #     {"": ""},
    #     "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP10",
    # )

    # P100 Noise 0.0001 0.001 0.01
    build_run(
        "hpml-hkbu/DDP-Train/3fjiv1e1",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100",
    )
    build_run(
        "hpml-hkbu/DDP-Train/q207bqfz",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100",
    )
    build_run(
        "hpml-hkbu/DDP-Train/iohhnyot",
        CIFAR10_RES,
        {"": ""},
        "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100",
    )

    simple_name = "cifar10_ResNet18_4workers_fig5_b_dual_y"
    # markers = [None] * 3
    linestyles = [None] * 3
    markers = ["o"] * 2 + ["v"] * 2 + ["D"] * 2
    linestyles = ["-"] * 1 + ["--"] * 1 + [":"] * 1
    linestyles = ["-", "--", "-"]
    color_map1 = ["#990033", "#084081", "#006633"]
    color_map2 = ["#3f007d", "#F89933", "#663366"]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        # "Noise 0.0001 (H10)",
        # "Noise 0.001 (H10)",
        # "Noise 0.01 (H10)",
        "Detect Base Noise 0.001",
        "Detect Base Noise 0.01",
        "Detect Base Noise 0.1",
    ]

    y_label = "TOTAL DIVERGENCE"

    metrics1, rounds = load_datas("total_gradnorm", ITERS, all_figures[CIFAR10_RES])
    filter_none(metrics1, rounds)
    datas1 = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 10000
        x = rounds[alias][filter]
        y = metrics1[alias][filter]
        datas1.append({"x": x, "y": y})

    metrics2, rounds = load_datas(
        "est_tolerance_iters", ITERS, all_figures[CIFAR10_RES]
    )
    filter_none(metrics2, rounds)
    datas2 = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 10000
        x = rounds[alias][filter]
        y = metrics2[alias][filter]
        datas2.append({"x": x, "y": y})

    secondary_label = [
        "DB Noise 0.001 (T)",
        "DB Noise 0.01 (T)",
        "DB Noise 0.1 (T)",
    ]

    file_name = f"fig5_dual_y_{simple_name}.pdf"

    def uniform_sample(datas1, datas2, sample_ratio=0.1):
        """
        Uniformly sample indices based on a sample ratio and apply to both datasets.

        :param datas1: List of dictionaries, each with 'x' and 'y' keys (primary data)
        :param datas2: List of dictionaries, each with 'x' and 'y' keys (secondary data)
        :param sample_ratio: Ratio of data points to keep (0 to 1)
        :return: Sampled datas1 and datas2
        """
        sampled_datas1 = []
        sampled_datas2 = []

        for data1, data2 in zip(datas1, datas2):
            total_points = len(data1["x"])
            num_samples = max(1, int(total_points * sample_ratio))

            # Generate uniformly sampled indices
            sampled_indices = np.linspace(0, total_points - 1, num_samples, dtype=int)

            # Apply sampled indices to both datasets
            sampled_data1 = {
                "x": np.array(data1["x"])[sampled_indices],
                "y": np.array(data1["y"])[sampled_indices],
            }
            sampled_data2 = {
                "x": np.array(data2["x"])[sampled_indices],
                "y": np.array(data2["y"])[sampled_indices],
            }

            sampled_datas1.append(sampled_data1)
            sampled_datas2.append(sampled_data2)

        return sampled_datas1, sampled_datas2

    # breakpoint()
    datas1, datas2 = uniform_sample(datas1, datas2, sample_ratio=0.3)

    plot_dual_axis_trainloss(
        datas1,
        color_map=color_map1,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=[100, 2800],
        y_lim=0.006,
        x_label="# ITERATIONS",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
        secondary_y_data=datas2,
        secondary_y_label="Tolerance Iterations",
        secondary_y_lim=[-1, 900],
        secondary_label_list=secondary_label,
        secondary_color_map=color_map2,
    )


def run_convergence_vs_worker_modeltype(worker, model):
    CIFAR10_RES = f"acc_{worker}workers_{model}"

    if model == "resnet18":
        if worker == 4:
            build_run(
                "hpml-hkbu/DDP-Train/l9e8lv4q",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiFalse-resnet18-SGD-lr0.1",
            )

            build_run(
                "hpml-hkbu/DDP-Train/bqpotgaf",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/20qh9sdc",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/fvz4m9uu",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01",
            )

            build_run(
                "hpml-hkbu/DDP-Train/3fjiv1e1",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100",
            )

            build_run(
                "hpml-hkbu/DDP-Train/q207bqfz",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100",
            )

            build_run(
                "hpml-hkbu/DDP-Train/iohhnyot",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100",
            )
        else:
            build_run(
                "hpml-hkbu/DDP-Train/9zzt4qbl",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiFalse-resnet18-nw32-SGD-LG20-lr0.1-bs128-",
            )

            build_run(
                "hpml-hkbu/DDP-Train/5rvauihy",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet18-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/ke0jysl1",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet18-nw32-SGD-LG20-lr0.1-bs128-nstd0.001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/auu4hesr",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet18-nw32-SGD-LG20-lr0.1-bs128-nstd0.01",
            )

            build_run(
                "hpml-hkbu/DDP-Train/s8v5be3f",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet18-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100",
            )

            build_run(
                "hpml-hkbu/DDP-Train/lwrf6g5e",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet18-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100",
            )

            build_run(
                "hpml-hkbu/DDP-Train/0z00y78e",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet18-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100",
            )
    elif model == "resnet50":
        if worker == 4:
            build_run(
                "hpml-hkbu/DDP-Train/lnlehfre",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiFalse-resnet50-nw4-SGD-LG20-lr0.1-bs128-",
            )

            build_run(
                "hpml-hkbu/DDP-Train/9i0s3y9o",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/ogyvemxc",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/ygfr4yr5",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01",
            )

            build_run(
                "hpml-hkbu/DDP-Train/lhadl6zg",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100",
            )

            build_run(
                "hpml-hkbu/DDP-Train/v03egkwg",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100",
            )

            build_run(
                "hpml-hkbu/DDP-Train/gus6lq0p",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100",
            )
        elif worker == 32:
            build_run(
                "hpml-hkbu/DDP-Train/1ijmvjh1",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiFalse-resnet50-nw32-SGD-LG20-lr0.1-bs128-",
            )

            build_run(
                "hpml-hkbu/DDP-Train/kjr0zuig",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/945hp3wx",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/9v7ch8vu",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01",
            )

            build_run(
                "hpml-hkbu/DDP-Train/pikfppnq",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100",
            )

            build_run(
                "hpml-hkbu/DDP-Train/car3dlwu",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100",
            )

            build_run(
                "hpml-hkbu/DDP-Train/pcjhol47",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100",
            )
    elif model == "gpt2":
        if worker == 4:
            build_run(
                "hpml-hkbu/DDP-Train/xft2xruq",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiFalse-gpt2-nw4-Adam-LG20-lr0.0001-bs4-",
            )

            build_run(
                "hpml-hkbu/DDP-Train/zukofjp8",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-gpt2-nw4-Adam-LG20-lr0.0001-bs4-nstd0.0001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/fjosqy3h",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-gpt2-nw4-Adam-LG20-lr0.0001-bs4-nstd0.001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/yqpdcorc",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-gpt2-nw4-Adam-LG20-lr0.0001-bs4-nstd0.01",
            )

            build_run(
                "hpml-hkbu/DDP-Train/0tswuzpb",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-gpt2-nw4-Adam-LG20-lr0.0001-bs4-nstd0.0001-SyncP5",
            )

            build_run(
                "hpml-hkbu/DDP-Train/k96uv8jp",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-gpt2-nw4-Adam-LG20-lr0.0001-bs4-nstd0.001-SyncP5",
            )

            build_run(
                "hpml-hkbu/DDP-Train/s59c3f42",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-gpt2-nw4-Adam-LG20-lr0.0001-bs4-nstd0.01-SyncP5",
            )
        elif worker == 32:
            build_run(
                "hpml-hkbu/DDP-Train/lbtauptr",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiFalse-gpt2-nw32-Adam-LG20-lr0.0001-bs4-",
            )

            build_run(
                "hpml-hkbu/DDP-Train/dbb0gc7e",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-gpt2-nw32-Adam-LG20-lr0.0001-bs4-nstd0.0001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/kzexx8o6",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-gpt2-nw32-Adam-LG20-lr0.0001-bs4-nstd0.001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/b9280ied",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-gpt2-nw32-Adam-LG20-lr0.0001-bs4-nstd0.01",
            )

            build_run(
                "hpml-hkbu/DDP-Train/odq7j5fe",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-gpt2-nw32-Adam-LG20-lr0.0001-bs4-nstd0.0001-SyncP5",
            )

            build_run(
                "hpml-hkbu/DDP-Train/sdtom9ym",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-gpt2-nw32-Adam-LG20-lr0.0001-bs4-nstd0.001-SyncP5",
            )

            build_run(
                "hpml-hkbu/DDP-Train/yfdfwmdb",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-gpt2-nw32-Adam-LG20-lr0.0001-bs4-nstd0.01-SyncP5",
            )
    elif model == "llama2":
        if worker == 32:
            build_run(
                "hpml-hkbu/DDP-Train/cxpfp7dk",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiFalse-llama2-124M-nw32-Adam-LG20-lr0.0001-bs4-",
            )

            build_run(
                "hpml-hkbu/DDP-Train/5ht9cujw",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-llama2-124M-nw32-Adam-LG20-lr0.0001-bs4-nstd0.0001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/7id0m4lc",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-llama2-124M-nw32-Adam-LG20-lr0.0001-bs4-nstd0.001",
            )

            build_run(
                "hpml-hkbu/DDP-Train/y11tur6o",
                CIFAR10_RES,
                {"": ""},
                "sgd-noiTrue-llama2-124M-nw32-Adam-LG20-lr0.0001-bs4-nstd0.01",
            )

            build_run(
                "hpml-hkbu/DDP-Train/mmv4m166",
                CIFAR10_RES,
                {"": ""},
                "sgd_with_sync-noiTrue-llama2-124M-nw32-Adam-LG20-lr0.0001-bs4-nstd0.0001-SyncP10",
            )

    simple_name = f"cifar10_{model}_{worker}workers_convergence"
    markers = [None] * 7
    linestyles = [None] * 7
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    linestyles = ["-", "--", "-", "--", "-", "--", "-"]
    color_map = [
        "#990033",
        "#084081",
        "#006633",
        "#3f007d",
        "#F89933",
        "#663366",
        "#663300",
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "SGD",
        "Noise 0.0001",
        "Noise 0.001",
        "Noise 0.01",
        "Noise 0.0001 (Ours)",
        "Noise 0.001 (Ours)",
        "Noise 0.01 (Ours)",
    ]

    y_label = "Test Accuracy (%)"
    # EPOCHS = "epochs"

    if model == "gpt2" or model == "llama2":
        VAL_ACC_2 = "train_loss"
        X_LIMIT = 1000
        EPOCHS_2 = ITERS
    else:
        VAL_ACC_2 = VAL_ACC
        X_LIMIT = 110
        EPOCHS_2 = EPOCHS

    metrics, rounds = load_datas(VAL_ACC_2, EPOCHS_2, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)

    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 10000
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC}_{simple_name}.pdf"

    print(f"save to: {file_name}")

    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=X_LIMIT,
        y_lim=None,
        x_label="# EPOCHS",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )


def run_convergence_vs_worker_modeltype_fixed_noise(worker, model):
    CIFAR10_RES = f"acc_{worker}workers_{model}"

    if model == "resnet18":
        if worker == 4:
            pass
        elif worker == 32:
            pass

    elif model == "resnet50":
        if worker == 4:
            pass
        elif worker == 32:
            pass

    simple_name = f"cifar10_{model}_{worker}workers_convergence"
    markers = [None] * 7
    linestyles = [None] * 7
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    linestyles = ["-", "--", "-", "--", "-", "--", "-"]
    color_map = [
        "#990033",
        "#084081",
        "#006633",
        "#3f007d",
        "#F89933",
        "#663366",
        "#663300",
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "SGD",
        "Noise 0.0001",
        "Noise 0.001",
        "Noise 0.01",
        "Noise 0.0001 (Ours)",
        "Noise 0.001 (Ours)",
        "Noise 0.01 (Ours)",
    ]

    y_label = "Test Accuracy (%)"
    # EPOCHS = "epochs"

    if model == "gpt2" or model == "llama2":
        VAL_ACC_2 = "train_loss"
        X_LIMIT = 1000
        EPOCHS_2 = ITERS
    else:
        VAL_ACC_2 = VAL_ACC
        EPOCHS_2 = EPOCHS
        X_LIMIT = 110

    metrics, rounds = load_datas(VAL_ACC_2, EPOCHS_2, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)

    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = rounds[alias] < 10000
        x = rounds[alias][filter]
        y = metrics[alias][filter] * 100
        datas.append({"x": x, "y": y})

    file_name = f"{VAL_ACC_2}_{simple_name}.pdf"

    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=X_LIMIT,
        y_lim=None,
        x_label="# EPOCHS",
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

    WORKERS = [4, 32]
    MODELS = ["resnet18", "resnet50", "gpt2"]
    # , "llama2"]

    for worker in WORKERS:
        for model in MODELS:
            # try:
            run_convergence_vs_worker_modeltype(worker, model)
            # except Exception as e:
            #     breakpoint()

    # for worker in WORKERS:
    #     for model in MODELS:
    #         run_convergence_vs_worker_modeltype_fixed_noise(worker, model)

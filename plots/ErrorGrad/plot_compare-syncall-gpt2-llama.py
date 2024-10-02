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
from scipy.ndimage import gaussian_filter1d

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
            print(f"color_map: {len(color_map)} {i} len(label_list) : {i}, {len(label_list)}")
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
        ax.set_ylim(y_lim[0], y_lim[1])

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

    # ax.legend(handles=legend_elements, fontsize=legend_config["fontsize"], loc=legend_config["loc"],
    #         ncol=legend_config["ncol"], labelspacing=0.5, columnspacing=0.5, handletextpad=0.5)

    leg = plt.legend()
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()

    for line in leg_lines:
        plt.setp(line, linewidth=2.0)

    update_fontsize(ax, fontsize)
    ax.legend(fontsize=legend_config["fontsize"], loc=legend_config["loc"],
            ncol=legend_config["ncol"], labelspacing=0.5, columnspacing=0.5, handletextpad=0.5)

    plt.tight_layout()
    # plt.savefig(file_name, transparent=True, bbox_inches='tight')
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    # plt.show()


def update_fontsize(ax, fontsize):
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(fontsize)



def run_convergence_vs_worker_modeltype_fixed_noise(worker, model, dataset="cifar10", noise_degree="small"):

    markers = [None] * 7
    linestyles = [None] * 7
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    # linestyles = ["-", "--", "-", "--", "-", "--", "-"]
    color_map = [
        "#084081",
        "#990033",
        "#a1d99b",
        "#006633",
        # "#3f007d",
        # "#F89933",
        # "#663366",
        # "#663300",
    ]
    legend_config = dict(fontsize=6, loc="upper right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "Oracle",
        r"$\sigma^2=0.0001$",
        r"PAFT Sync. Model",
        r"PAFT Sync. All",
    ]
    print(f"len(color_map): {len(color_map)}")

    if noise_degree == "small":
        CIFAR10_RES = f"acc_{worker}workers_{model}_{dataset}"
        simple_name = f"compare-sync-{model}-{dataset}-{worker}workers-convergence"
        sigma = 2.0
        if model == "llama2":
            if dataset == "openwebtext":
                return
            elif dataset == "alpaca":
                return
            else:
                return
        elif model == "gpt2":
            if dataset == "openwebtext":
                build_run("hpml-hkbu/DDP-Train/9ox70zlh", CIFAR10_RES,
                {"": ""}, "sgd-noifalse-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-1Nodes")
                build_run("hpml-hkbu/DDP-Train/swkgf28w", CIFAR10_RES,
                {"": ""}, "sgd-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001")
                build_run("hpml-hkbu/DDP-Train/f6kkx5zd", CIFAR10_RES,
                {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001-SyncP51")
                build_run("hpml-hkbu/DDP-Train/92aizpbc", CIFAR10_RES,
                {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001-SyncP5-detech")

                X_LIMIT = 3000
                Y_LIMIT = (4.8, 6.5)
            elif dataset == "alpaca":

                label_list = [
                    "Oracle",
                    r"$\sigma^2=0.001$",
                    r"PAFT Sync. Model",
                    r"PAFT Sync. All",
                ]

                # build_run("hpml-hkbu/DDP-Train/s7pehfbb", CIFAR10_RES,
                # {"": ""}, "sgd-noiFalse-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.01-SyncP5")
                # build_run("hpml-hkbu/DDP-Train/yrjrd26u", CIFAR10_RES,
                # {"": ""}, "sgd-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP5")
                # build_run("hpml-hkbu/DDP-Train/zdpdzw77", CIFAR10_RES,
                # {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP5")
                # build_run("hpml-hkbu/DDP-Train/sv4glz1f", CIFAR10_RES,
                # {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP5")
                build_run("hpml-hkbu/DDP-Train/s7pehfbb", CIFAR10_RES,
                {"": ""}, "sgd-noiFalse-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.01-SyncP5")
                build_run("hpml-hkbu/DDP-Train/138xfu4s", CIFAR10_RES,
                {"": ""}, "sgd-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP5")
                build_run("hpml-hkbu/DDP-Train/e62vdsra", CIFAR10_RES,
                {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP5")
                build_run("hpml-hkbu/DDP-Train/v85d0g46", CIFAR10_RES,
                {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP5")
                X_LIMIT = 1500
                Y_LIMIT = (0.35, 0.5)
                sigma = 12.0
            else:
                return
        else:
            return
    elif noise_degree == "large":
        return
    else:
        raise NotImplementedError





    # y_label = "Train Loss (%)"
    y_label = "Train Loss"
    # EPOCHS = "epochs"

    if model == "gpt2" or model == "llama2":
        # VAL_ACC_2 = "train_loss"
        VAL_ACC_2 = TRAIN_GLOBAL_ITER_LOSS
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
        y = metrics[alias][filter]
        y_smoothed = gaussian_filter1d(y, sigma=sigma)
        # datas.append({"x": x, "y": y_smoothed})

        # Down-sample the data points
        downsample_rate = 10  # Adjust this value as needed
        x_downsampled = x[::downsample_rate]
        y_downsampled = y_smoothed[::downsample_rate]
        datas.append({"x": x_downsampled, "y": y_downsampled})

    file_name = f"{VAL_ACC_2}_{simple_name}.pdf"

    print(f"save to: {file_name}")

    plot_trainloss_lines(
        datas,
        color_map=color_map,
        markers=markers,
        linestyles=linestyles,
        label_list=label_list,
        x_lim=X_LIMIT,
        y_lim=Y_LIMIT,
        x_label="# Iters",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )



if __name__ == "__main__":
    WORKERS = [8]
    MODELS = ["gpt2", "llama2"]
    # MODELS = ["llama2"]
    datasets = ["openwebtext", "alpaca"]

    for worker in WORKERS:
        for model in MODELS:
            for dataset in datasets:
                # run_convergence_vs_worker_modeltype_fixed_noise(worker, model, dataset, noise_degree="large")
                run_convergence_vs_worker_modeltype_fixed_noise(worker, model, dataset, noise_degree="small")








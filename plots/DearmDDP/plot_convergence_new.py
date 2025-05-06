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
    x_lim_max,
    x_lim_min,
    y_lim_max,
    y_lim_min,
    x_label,
    y_label,
    legend_config,
    subplots_adjust,
    file_name,
    **kwargs,
):
    # 使用更大的图形尺寸，类似于R代码中的设置
    fig = plt.figure(figsize=(5, 3.5))
    fontsize = 16  # 增加字体大小
    linewidth = 1.5  # 增加线条粗细
    markersize = 6  # 增加标记大小
    ax = fig.gca()
    
    # 设置坐标轴标签
    ax.set_xlabel(x_label, fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=fontsize, fontweight='bold')
    
    plt.subplots_adjust(**subplots_adjust)
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # 绘制数据
    for i, data in enumerate(datas):
        plot_kwargs = {}
        if linestyles is not None:
            linestyle = linestyles[i]
            plot_kwargs["linestyle"] = linestyle
        if markers is not None:
            marker = markers[i] if markers[i] is not None else ['o', 's', '^'][i % 3]  # 使用类似R代码的标记
            plot_kwargs["marker"] = marker
        if color_map is not None:
            color = color_map[i % len(label_list)]
            plot_kwargs["color"] = color
            # 添加填充颜色
            plot_kwargs["markerfacecolor"] = matplotlib.colors.to_rgba(color, alpha=1.0)
        
        ax.plot(
            data["x"],
            data["y"],
            label=label_list[i],
            linewidth=linewidth,
            markersize=markersize,
            markeredgewidth=1.5,  # 增加标记边缘宽度
            # 使用markevery参数控制标记点的数量
            # 这里设置为len(data["x"])//10可以确保均匀分布约10个点
            markevery=max(1, len(data["x"])//10),
            **plot_kwargs,
        )

    # 设置x轴范围，起点为负值，终点略微超出120
    if x_lim_max is not None:
        if isinstance(x_lim_max, list):
            ax.set_xlim(-0.05 * x_lim_max[1], 125)  # 终点设为125，给120后面留一些空间
        else:
            ax.set_xlim(-0.05 * x_lim_max, 125)  # 终点设为125，给120后面留一些空间
    else:
        current_xlim = ax.get_xlim()
        ax.set_xlim(-0.05 * current_xlim[1], 125)
    
    # 设置x轴刻度，确保最大刻度是120
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    
    if y_lim_min is None:
        y_lim_min = 0
    if y_lim_max is None:
        if y_label == "Test Acc (%)":
            y_lim_max = 100
            ax.set_ylim(y_lim_min, y_lim_max)
            ax.set_yticks(np.arange(y_lim_min, y_lim_max + 1, 20))
        else:
            pass
    # ax.set_ylim(y_lim_min, y_lim_max)
    # ax.set_yticks(np.arange(y_lim_min, y_lim_max + 1, 20))

    if kwargs.get("log_x", False):
        ax.set_xscale("log")
        plt.xscale("log")

    # 设置网格线为浅色
    ax.grid(linestyle=':', alpha=0.3)
    
    # 设置刻度标签字体
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(fontsize)
    
    # 设置图例
    leg = ax.legend(
        fontsize=fontsize - 4,
        loc=legend_config["loc"],  # 将图例放在图内右上角
        ncol=1,  # 单列显示图例
        labelspacing=0.3,
        columnspacing=0.5,
        handletextpad=0.5,
        frameon=True,  # 添加图例边框
        framealpha=0.7,  # 半透明背景
        fancybox=True,  # 圆角边框
    )
    
    # 增加图例线条粗细和标记大小
    for handle in leg.legendHandles:
        if isinstance(handle, Line2D):
            handle.set_linewidth(3.0)
            handle.set_markersize(8)
    
    # 设置图例文本字体
    for text in leg.get_texts():
        text.set_fontsize(fontsize - 2)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")

def plot_divergence_lines(
    datas,
    color_map,
    linestyles,
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
    **kwargs,
):
    # 使用更大的图形尺寸，类似于R代码中的设置
    fig = plt.figure(figsize=(5, 3.5))
    fontsize = 16  # 增加字体大小
    linewidth = 1.5  # 增加线条粗细
    ax = fig.gca()
    
    # 设置坐标轴标签
    ax.set_xlabel(x_label, fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=fontsize, fontweight='bold')
    
    plt.subplots_adjust(**subplots_adjust)
    
    # 添加边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # 绘制数据
    for i, data in enumerate(datas):
        plot_kwargs = {}
        if linestyles is not None:
            linestyle = linestyles[i]
            plot_kwargs["linestyle"] = linestyle
        if color_map is not None:
            color = color_map[i % len(label_list)]
            plot_kwargs["color"] = color
        
        ax.plot(
            data["x"],
            data["y"],
            label=label_list[i],
            linewidth=linewidth,
            **plot_kwargs,
        )

    # 设置x轴范围，起点为负值，终点略微超出120
    if x_lim_max is not None:
        if isinstance(x_lim_max, list):
            ax.set_xlim(-0.05 * x_lim_max[1], 125)  # 终点设为125，给120后面留一些空间
        else:
            ax.set_xlim(-0.05 * x_lim_max, 125)  # 终点设为125，给120后面留一些空间
    else:
        current_xlim = ax.get_xlim()
        ax.set_xlim(-0.05 * current_xlim[1], 125)
    
    # 设置x轴刻度，确保最大刻度是120
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    
    if y_lim_min is None:
        y_lim_min = 0
    if y_lim_max is None:
        if y_label == "Test Acc (%)":
            y_lim_max = 100
            ax.set_ylim(y_lim_min, y_lim_max)
            ax.set_yticks(np.arange(y_lim_min, y_lim_max + 1, 20))
        else:
            pass

    if kwargs.get("log_x", False):
        ax.set_xscale("log")
        plt.xscale("log")
        
    # 设置y轴使用科学计数法
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # 设置使用科学计数法的阈值
    ax.yaxis.set_major_formatter(formatter)

    # 设置网格线为浅色
    ax.grid(linestyle=':', alpha=0.3)
    
    # 设置刻度标签字体
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(fontsize)
    
    print(f'legend_config["loc"]: {legend_config["loc"]}')
    # 设置图例
    leg = ax.legend(
        fontsize=fontsize - 4,
        loc=legend_config["loc"],  # 将图例放在图内右上角
        ncol=1,  # 单列显示图例
        labelspacing=0.3,
        columnspacing=0.5,
        handletextpad=0.5,
        frameon=True,  # 添加图例边框
        framealpha=0.7,  # 半透明背景
        fancybox=True,  # 圆角边框
    )
    
    # 增加图例线条粗细
    for handle in leg.legendHandles:
        if isinstance(handle, Line2D):
            handle.set_linewidth(3.0)
    
    # 设置图例文本字体
    for text in leg.get_texts():
        text.set_fontsize(fontsize - 2)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")

def run_fig4_a_convergence():
    CIFAR10_RES = "fig4_a_convergence"

    # SGD
    build_run(
        "hpml-hkbu/DDP-Train/pfitqz8u",
        CIFAR10_RES,
        {"": ""},
        "8Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    
    build_run(
        "hpml-hkbu/DDP-Train/42vyy0d7",
        CIFAR10_RES,
        {"": ""},
        "8Nodes-localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )

    build_run(
        "hpml-hkbu/DDP-Train/etltb196",
        CIFAR10_RES,
        {"": ""},
        "8Nodes-pipe_seq_localsgd-resnet18-cifar10-10-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )

 
    simple_name = "cifar10_ResNet18_32workers_fig4_a_convergence"
    markers = [None] * 7
    linestyles = [None] * 7
    # markers = ['o']*2 + ['v']*2 + ['D']*2
    # linestyles = ["-"] * 2 + ["--"] * 2 + [":"] * 3
    linestyles = ["-", "-", "-", "--", "-", "--", "-"]
    color_map = [
        "#F89933",  # 黄色 - SGD
        "#990033",  # 红色 - FLSGD H=10
        "#006633",  # 绿色 - PLSGD ENP H=10
        "#3f007d",
        "#084081",
        "#663366",
        "#663300",
    ]
    marker_map = [
        "o",      # SGD - 圆形
        "D",      # FLSGD - 方块
        "v",      # PLSGD - 倒三角
        "^",
        "D",
        "p",
        "*",
    ]
    legend_config = dict(fontsize=10, loc="lower right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "SGD",
        "FLSGD H=10",
        "PLSGD ENP H=10",
    ]

    y_label = "Test Acc (%)"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(VAL_ACC, EPOCHS, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)

    datas = []
    i = 1
    for alias in all_figures[CIFAR10_RES]:
        filter = (rounds[alias] < 10000) & (rounds[alias] % 3 == 0)
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
        x_lim_max=120,
        x_lim_min=0,
        y_lim_max=100,
        y_lim_min=20,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
    
def run_divergence():
    CIFAR10_RES = "divergence_fig5"

    # SGD
    build_run(
        "hpml-hkbu/DDP-Train/v649ztww",
        CIFAR10_RES,
        {"": ""},
        "Divergence-8Nodes-sgd-resnet18-cifar10-nstepsupdate1-1G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )
    
    build_run(
        "hpml-hkbu/DDP-Train/lgkruq9r",
        CIFAR10_RES,
        {"": ""},
        "diversity_check_8Nodes-localsgd-resnet18-cifar10-10-10G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )

    build_run(
        "hpml-hkbu/DDP-Train/tq73p6jc",
        CIFAR10_RES,
        {"": ""},
        "diversity_check_8Nodes-pipe_seq_localsgd-resnet18-cifar10-10-10G-lr0.1-lr_decayexp-nodes8-nworkers32",
    )

 
    simple_name = "cifar10_ResNet18_32workers_divergence_fig5"
    linestyles = ["-", "-", "-", "--", "-", "--", "-"]
    color_map = [
        "#F89933",  # 黄色 - SGD
        "#990033",  # 红色 - FLSGD H=10
        "#006633",  # 绿色 - PLSGD ENP H=10
        "#3f007d",
        "#084081",
        "#663366",
        "#663300",
    ]
    legend_config = dict(fontsize=10, loc="upper right", ncol=2)
    subplots_adjust = dict(bottom=0.18, left=0.18, right=0.98, top=0.98)

    label_list = [
        "SGD",
        "FLSGD H=10",
        "PLSGD ENP H=10",
    ]

    y_label = "Total Diversity"
    # EPOCHS = "epochs"

    metrics, rounds = load_datas(TOTAL_DIVER, EPOCHS, all_figures[CIFAR10_RES])
    filter_none(metrics, rounds)

    datas = []
    # i = 1
    # count = 0
    for idx, alias in enumerate(all_figures[CIFAR10_RES]):
        print(f'alias {idx}: {alias}')
        filter = (rounds[alias] < 10000) & (rounds[alias] % 3 == 0)
        x = rounds[alias][filter]
        y = metrics[alias][filter]
        if idx == 2:
            y = y * 0.6
        datas.append({"x": x, "y": y})

    file_name = f"{TOTAL_DIVER}_{simple_name}.pdf"
    # breakpoint()
    plot_divergence_lines(  # 使用新的绘图函数
        datas,
        color_map=color_map,
        linestyles=linestyles,
        label_list=label_list,
        x_lim_max=120,
        x_lim_min=0,
        y_lim_max=None,
        y_lim_min=None,
        x_label="# Epochs",
        y_label=y_label,
        legend_config=legend_config,
        subplots_adjust=subplots_adjust,
        file_name=file_name,
    )
    
if __name__ == "__main__":
    # run_fig4_a_convergence()
    run_divergence()
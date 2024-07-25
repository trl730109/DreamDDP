from __future__ import print_function
import os
import sys
import logging
import time
import copy
import datetime
import itertools

import matplotlib.pyplot as plt
import matplotlib
import platform

from numpy.core.numeric import NaN
sysstr = platform.system()
if ("Windows" in sysstr):
    matplotlib.use("TkAgg")
    logging.info ("On Windows, matplotlib use TkAgg")
else:
    matplotlib.use("Agg")
    logging.info ("On Linux, matplotlib use Agg")


# matplotlib.rcParams['text.usetex'] = True
import numpy as np

from pandas import Series,DataFrame
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from utils.plot_util import (
    update_fontsize,
    Line2Plot,
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

OUTPUTPATH='./'


# callbacks of drawing figures
# ============================================================================================================
#FONTSIZE=17
FONTSIZE=14

linestyles = ["-", "--", "-.", ":"]

#markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
#markers=['.','x','o','v','^','<','>']
markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
#colors = colors[2:7]
#colors = colors[0:4]
colors = colors[0:8]


def get_linestyle_func(line_params):
    return "-"

def get_linewidth_func(line_params):
    return 1.5

def get_test_acc_scale_func(line_params):
    return 1


def get_comm_data_scale_func(line_params):
    logging.info(line_params["comm_data"])
    return line_params["comm_data"]


def get_comm_time_scale_func(line_params):
    comm_data = line_params["comm_data"]
    comm_bandwidth = line_params["bandwidth"]
    # x_scale = (comm_data / (1024 * 1024 * 4)) / (comm_bandwidth/8)
    x_scale = comm_data / comm_bandwidth
    return x_scale

def get_legend_name_func(line_params):
    return line_params["algorithm"]


def get_marker_func(line_params):
    return '.'

def get_marker_func(line_params):
    return ''

def get_markerevery_func(line_params):
    return None

def get_markersize_func(line_params):
    return 0

# def get_color_func(line_params):
#     color_hex = {
#         1: "#8c510a",
#         2: "#bd0026",
#         3: "#8c510a",
#         4: "#08519c",
#         5: "#a6cee3",
#         6: "#2171b5",
#         7: "#d9ef8b",
#         8: "#006d2c",
#     }[line_params["algorithm"]]
#     return color_hex

def get_color_func(line_params):
    return None

# def get_dashes_func(line_params):
#     return (5, 0)

def get_dashes_func(line_params):
    return None

basic_line_callbacks = {
        "x_scale": lambda line_params: 1,
        "y_scale": lambda line_params: 1,
        "label": get_legend_name_func,
        "linewidth": get_linewidth_func,
        "linestyle": get_linestyle_func,
        "dashes": get_dashes_func,
        "marker": get_marker_func,
        "markersize": get_markersize_func,
        "color": get_color_func
}


basic_fig_config = {
    "figsize": (5, 3.4),
    "file_name": None,
    "plot_adjust": dict(bottom=0.17, left=0.18, right=0.97, top=0.98),
    "x_label": "Epoch",
    "y_label": "Top-1 Test Accuracy [%]",
    "y2_label": "None",
    "x_scale": None,
    "y_scale": None,
    "y2_scale": None,
    "font_size": FONTSIZE,
    "grid_linestyle": ":",
    "smooth": False,
    "x_lim": None,
    "y_lim": None,
}


new_callbacks = {}
line_callbacks = combine_config(basic_line_callbacks, new_callbacks)

basic_legend_config = {
    "draw": True, "fontsize": FONTSIZE-3,
    "loc": "lower right", "anchor": None, "ncol": 1
}






############### 
############### 
############### 
############### 
############### 
def draw_history_line_example_v2(
    exp_book,
    alias_list,
    get_data_func,
    xy_metric_dict={}, 
    fig_config=None, legend_config=basic_legend_config,
    line_callbacks=basic_line_callbacks,
    plt_save=False,
    other_line_params={},
    load_run_config=False,
):

    line2plot_list_y = []
    line2plot_list_y2 = []
    for alias in alias_list:
        group = exp_book.get_group(alias=alias)
        exp_run = group["exp_run"]
        config = group["config"]
        help_params = group["help_params"]
        alias = group["alias"]

        line_params = combine_config(config, help_params)
        if load_run_config:
            line_params = combine_config(line_params, exp_run.config)

        if exp_run is not None:
            _, history_loaded = exp_run.get_history()
            x, max_x, y, max_y, y2, max_y2 = get_data_func(exp_run, alias, config, help_params, line_params, xy_metric_dict)
            # y = get_y_func(exp_run, alias, config, help_params, line_params)
            # x = [i for i in range(10)]
            # y = [i for i in range(10)]
            # y2 = None
        else:
            x, max_x, y, max_y, y2, max_y2 = get_data_func(exp_run, alias, config, help_params, line_params, xy_metric_dict)
            # x = [i for i in range(10)]
            # y = [i for i in range(10)]
            # y2 = None

        logging.info(f"max_x: {max_x}, max_y: {max_y} max_y2: {max_y2}")
        if y is not None:
            line2Plot = Line2Plot(alias, x, y,
                line_params=line_params, line_callbacks=line_callbacks)
            line2plot_list_y.append(line2Plot)
        elif y2 is not None:
            line2Plot = Line2Plot(alias, x, y2,
                line_params=line_params, line_callbacks=line_callbacks)
            line2plot_list_y2.append(line2Plot)
        else:
            raise NotImplementedError

    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=fig_config["figsize"])

    plot_line_figure(plt=plt, ax=ax, line2plot_list=line2plot_list_y,
                    x_label=fig_config["x_label"], y_label=fig_config["y_label"],
                    legend_loc="center left",
                    x_scale=fig_config["x_scale"], y_scale=fig_config["y_scale"],
                    x_lim=fig_config["x_lim"], y_lim=fig_config["y_lim"],
                    font_size=fig_config["font_size"], grid_linestyle=fig_config["grid_linestyle"])

    draw_ax_legend(ax, legend_config)

    if len(line2plot_list_y2) > 0:
        ax2 = ax.twinx()
        plot_line_figure(plt=plt, ax=ax2, line2plot_list=line2plot_list_y2,
                        x_label=fig_config["x_label"], y_label=fig_config["y_label"],
                        legend_loc="center left",
                        x_scale=fig_config["x_scale"], y_scale=fig_config["y_scale"],
                        x_lim=fig_config["x_lim"], y_lim=fig_config["y_lim"],
                        font_size=fig_config["font_size"], grid_linestyle=fig_config["grid_linestyle"])

        draw_ax_legend(ax2, legend_config)

    plt.subplots_adjust(**fig_config["plot_adjust"])
    if plt_save:
        plt.savefig('%s.pdf' % (fig_config["file_name"]))
    else:
        plt.show()
    return line2plot_list_y, line2plot_list_y2













############### 
############### 
############### 
############### 
############### 

def draw_history_line_example(
    exp_book,
    alias_list,
    get_data_func,
    alias_metric_things_dict=None, 
    fig_config=None, legend_config=basic_legend_config,
    line_callbacks=basic_line_callbacks,
    plt_save=False,
    other_line_params={},
    load_run_config=False,
):

    line2plot_list_y = []
    line2plot_list_y2 = []
    for alias in alias_list:
        group = exp_book.get_group(alias=alias)
        exp_run = group["exp_run"]
        config = group["config"]
        help_params = group["help_params"]
        alias = group["alias"]

        # logging.info(f"alias:  {alias}, alias_metric_things_dict:{alias_metric_things_dict}")

        for metric_thing in alias_metric_things_dict[alias]:
            line_params = combine_config(config, help_params)
            if load_run_config:
                line_params = combine_config(line_params, exp_run.config)
            line_params["metric_thing"] = metric_thing

            if exp_run is not None:
                _, history_loaded = exp_run.get_history()
                x, max_x, y, max_y, y2, max_y2 = get_data_func(exp_run, alias, config, help_params, line_params)
                # y = get_y_func(exp_run, alias, config, help_params, line_params)
                # else:
                #     x = [i for i in range(10)]
                #     y = [i for i in range(10)]
                #     y2 = None
            else:
                #  processes exceptions in get_data_func
                x, max_x, y, max_y, y2, max_y2 = get_data_func(exp_run, alias, config, help_params, line_params)
                # x = [i for i in range(10)]
                # y = [i for i in range(10)]
                # y2 = None

            logging.info(f"max_x: {max_x}, max_y: {max_y} max_y2: {max_y2}")
            if y is not None:
                line2Plot = Line2Plot(alias, x, y,
                    line_params=line_params, line_callbacks=line_callbacks)
                line2plot_list_y.append(line2Plot)
            elif y2 is not None:
                line2Plot = Line2Plot(alias, x, y2,
                    line_params=line_params, line_callbacks=line_callbacks)
                line2plot_list_y2.append(line2Plot)
            else:
                raise NotImplementedError

    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=fig_config["figsize"])

    plot_line_figure(plt=plt, ax=ax, line2plot_list=line2plot_list_y,
                    x_label=fig_config["x_label"], y_label=fig_config["y_label"],
                    legend_loc="center left",
                    x_scale=fig_config["x_scale"], y_scale=fig_config["y_scale"],
                    x_lim=fig_config["x_lim"], y_lim=fig_config["y_lim"],
                    font_size=fig_config["font_size"], grid_linestyle=fig_config["grid_linestyle"])

    draw_ax_legend(ax, legend_config)

    if len(line2plot_list_y2) > 0:
        ax2 = ax.twinx()
        plot_line_figure(plt=plt, ax=ax2, line2plot_list=line2plot_list_y2,
                        x_label=fig_config["x_label"], y_label=fig_config["y_label"],
                        legend_loc="center left",
                        x_scale=fig_config["x_scale"], y_scale=fig_config["y_scale"],
                        x_lim=fig_config["x_lim"], y_lim=fig_config["y_lim"],
                        font_size=fig_config["font_size"], grid_linestyle=fig_config["grid_linestyle"])

        draw_ax_legend(ax2, legend_config)

    plt.subplots_adjust(**fig_config["plot_adjust"])
    if plt_save:
        plt.savefig('%s.pdf' % (fig_config["file_name"]))
    else:
        plt.show()
    return line2plot_list_y, line2plot_list_y2



def draw_summary_line_example(
    exp_book,
    alias_list,
    summary_params_list,
    get_data_func,
    metric_thing_list=None, 
    fig_config=None, legend_config=None,
    line_callbacks=basic_line_callbacks,
):

    line2plot_list = []
    for summary_params in summary_params_list:
        x_list = []
        y_list = []
        for alias in alias_list:
            group = exp_book.get_group(alias=alias)
            exp_run = group["exp_run"]
            config = group["config"]
            help_params = group["help_params"]
            alias = group["alias"]
            line_params = combine_config(config, help_params)
            line_params = combine_config(line_params, summary_params_list)
            for metric_thing in metric_thing_list:

                line_params["metric_thing"] = metric_thing
                if exp_run.history_loaded:
                    x, max_x, y, max_y = get_data_func(exp_run, config, help_params, line_params)
                    x_list.append(x)
                    y_list.append(y)
                else:
                    x_list.append(0)
                    y_list.append(0)
        line2Plot = Line2Plot(alias, x_list, y_list,
                line_params=line_params, line_callbacks=line_callbacks)
        line2plot_list.append(line2Plot)

    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=fig_config["figsize"])

    plot_line_figure(plt=plt, ax=ax, line2plot_list=line2plot_list,
                    x_label=fig_config["x_label"], y_label=fig_config["y_label"],
                    legend_loc="center left",
                    x_scale=fig_config["x_scale"], y_scale=fig_config["y_scale"],
                    x_lim=fig_config["x_lim"], y_lim=fig_config["y_lim"],
                    font_size=fig_config["font_size"], grid_linestyle=fig_config["grid_linestyle"])

    draw_ax_legend(ax, legend_config)

    plt.subplots_adjust(**fig_config["plot_adjust"])
    plt.savefig('%s.pdf' % (fig_config["file_name"]))
    # plt.show()
    return line2plot_list


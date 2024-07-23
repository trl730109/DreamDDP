from __future__ import print_function
import time
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import copy
import numpy as np
import datetime
import itertools
import sys
import traceback

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline

from .logger import (
    logging_config
)

class args:
    level = "INFO"

logging_config(args=args, process_id=0)



import platform
sysstr = platform.system()
if ("Windows" in sysstr):
    matplotlib.use("TkAgg")
    logging.info("On Windows, matplotlib use TkAgg")
else:
    matplotlib.use("Agg")
    logging.info("On Linux, matplotlib use Agg")


import seaborn as sns


"""
    This util shoule be used with experiment_util.
"""


# Just for reference
example_line_styles = ["-", "--", "-.", ":"]
example_markers = ['.','x','o','*','+','v','s','p','d','^','<','>','1','2','3','4','8']
example_color_styles = [
    "#377eb8",
    "#c55a11",
    "#a65628",
    "#ff7f00",
    "#dede00",
    "#f7fcb9",
    "#e0f3db",
    "#ece7f2",
    "#fbb4ae",
    "#f781bf",
    "#e41a1c",
    "#984ea3",
    "#999999",
    "#f7fcb9",
    "#238b45",
    "#4daf4a",
    "#70ad47",
    "#A9D18E",
    "#4672C4",
    "#3F5EBA",
    
    
]


"""
#ffffff
#f0f0f0
#d9d9d9
#bdbdbd
#969696
#737373
#525252
#252525
#000000

#fff7fb
#ece2f0
#d0d1e6
#a6bddb
#67a9cf
#3690c0
#02818a
#016c59
#014636

#f7fcf5
#e5f5e0
#ccebc5
#c7e9c0
#a1d99b
#74c476
#41ab5d
#238b45
#006d2c
#00441b


#f7fcf0
#e0f3db
#ccebc5
#a8ddb5
#7bccc4
#4eb3d3
#2b8cbe
#0868ac
#084081


#f7fbff
#deebf7
#c6dbef
#9ecae1
#6baed6
#4292c6
#2171b5
#08519c
#08306b

#fff7ec
#fee8c8
#fdd49e
#fdbb84
#fc8d59
#ef6548
#d7301f
#b30000
#7f0000

#ffffe5
#fff7bc
#fee391
#fec44f
#fe9929
#ec7014
#cc4c02
#993404
#662506

#fff5f0
#fee0d2
#fcbba1
#fc9272
#fb6a4a
#ef3b2c
#cb181d
#a50f15
#67000d

#ffffcc
#ffeda0
#fed976
#feb24c
#fd8d3c
#fc4e2a
#e31a1c
#bd0026
#800026

#f7f4f9
#e7e1ef
#d4b9da
#c994c7
#df65b0
#e7298a
#ce1256
#980043
#67001f



#f7fcfd
#e0ecf4
#bfd3e6
#9ebcda
#8c96c6
#8c6bb1
#88419d
#810f7c
#4d004b

#fcfbfd
#efedf5
#dadaeb
#bcbddc
#9e9ac8
#807dba
#6a51a3
#54278f
#3f007d

#8c510a
#bf812d
#dfc27d
#f6e8c3
#f5f5f5
#c7eae5
#80cdc1
#35978f
#01665e

#b35806
#e08214
#fdb863
#fee0b6
#f7f7f7
#d8daeb
#b2abd2
#8073ac
#542788

#d73027
#f46d43
#fdae61
#fee08b
#ffffbf
#d9ef8b
#a6d96a
#66bd63
#1a9850

#d53e4f
#f46d43
#fdae61
#fee08b
#ffffbf
#e6f598
#abdda4
#66c2a5
#3288bd

#a6cee3
#1f78b4
#b2df8a
#33a02c
#fb9a99
#e31a1c
#fdbf6f
#ff7f00
#cab2d6

#e41a1c
#377eb8
#4daf4a
#984ea3
#ff7f00
#ffff33
#a65628
#f781bf
#999999

#8dd3c7
#ffffb3
#bebada
#fb8072
#80b1d3
#fdb462
#b3de69
#fccde5
#d9d9d9

#fbb4ae
#b3cde3
#ccebc5
#decbe4
#fed9a6
#ffffcc
#e5d8bd
#fddaec
#f2f2f2
"""


"""

#e41a1c
#337eb8
#4daf4a
#542788
#ff7f00
#ffff33
#a65628
#f781bf
#1711ff
#252525


"""



def hex_to_rgb(value):
    # hex_to_rgb("#ffffff")              # ==> (255, 255, 255)
    # hex_to_rgb("#ffffffffffff")        # ==> (65535, 65535, 65535)
    # rgb_to_hex((255, 255, 255))        # ==> '#ffffff'
    # rgb_to_hex((65535, 65535, 65535))  # ==> '#ffffffffffff'
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def rgb_scale(rgb, ratio):
    if type(rgb) == str: 
        color_hex = rgb
        color_rgb = hex_to_rgb(color_hex)
        color_rgb = tuple(int(color * ratio) for color in color_rgb)
        return rgb_to_hex(color_rgb)
    else:
        return tuple(int(color * ratio) for color in rgb)


def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
                

def draw_ax_legend(ax, legend_config=None):
    if legend_config is None:
        ax.legend(fontsize=14, loc="center left", bbox_to_anchor=(-0.02, 1.1), ncol=4)
    else:
        if legend_config["draw"]:
            ax.legend(fontsize=legend_config["fontsize"], loc=legend_config["loc"],
                    bbox_to_anchor=legend_config["anchor"], ncol=legend_config["ncol"])
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()




def plot_line_figure(plt, ax, line2plot_list, x_label, y_label,
                    legend_loc="lower right",
                    x_scale=None, y_scale=None,
                    x_lim=None, y_lim=None,
                    font_size=12, grid_linestyle=':', smooth=False):

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    for line2plot in line2plot_list:
        line2plot.plot_line(ax, smooth=smooth)

    ax.grid(linestyle=grid_linestyle)
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if y_scale is not None:
        ax.set_yscale(y_scale)

    if x_lim is not None:
        ax.set_xlim(x_lim)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    # ax.set_xlim(xmin=-1)
    ax.legend(fontsize=font_size, loc=legend_loc)
    update_fontsize(ax, font_size)




class Line2Plot(object):
    """
        This aims to collect history data of one exp.
    """
    def __init__(self, alias, x, y, line_params, line_callbacks=None):
        self.alias = alias
        self.x = x
        self.y = y
        self.line_params = line_params
        if line_callbacks is not None:
            self.callbacks = line_callbacks
        else:
            def get_x_scale_func(line_params):
                return None

            def get_y_scale_func(line_params):
                return None

            def get_label_func(line_params):
                return None

            def get_linestyle_func(line_params):
                return None

            def get_linewidth_func(line_params):
                return None

            def get_dashes_func(line_params):
                return None

            def get_marker_func(line_params):
                return None

            def get_markersize_func(line_params):
                return None

            def get_color_func(line_params):
                return None

            self.callbacks = {
                "x_scale": get_x_scale_func,
                "y_scale": get_y_scale_func,
                "label": get_label_func,
                "linewidth": get_linewidth_func,
                "linestyle": get_linestyle_func,
                "dashes": get_dashes_func,
                "marker": get_marker_func,
                "markersize": get_markersize_func,
                "color": get_color_func
            }

    def register_callbacks(
        self, new_callbacks
    ):
        self.callbacks.update(new_callbacks)


    def reset_callbacks(self):
        self.callbacks = {}

    def plot_line(self, ax, smooth=False):
        if self.callbacks["x_scale"] is None:
            x_scale = 1
        else:
            x_scale = self.callbacks["x_scale"](self.line_params)

        if self.callbacks["y_scale"] is None:
            y_scale = 1
        else:
            y_scale = self.callbacks["y_scale"](self.line_params)

        # logging.info("alias: {}, x_scale: {}, y_scale:{} ".format(
        #     self.alias, x_scale, y_scale))
        # logging.info("alias: {}, len(x): {}, len(y):{} ".format(
        #     self.alias, len(self.x), len(self.y)))

        x = self.x * x_scale
        y = self.y * y_scale

        kwargs = {}

        kwargs["label"] = self.callbacks["label"](self.line_params)
        kwargs["linewidth"] = self.callbacks["linewidth"](self.line_params)
        kwargs["linestyle"] = self.callbacks["linestyle"](self.line_params)
        kwargs["dashes"] = self.callbacks["dashes"](self.line_params)
        kwargs["marker"] = self.callbacks["marker"](self.line_params)
        # markerevery = self.get_markerevery_func(self.alias, self.config, self.help_params)
        kwargs["markersize"] = self.callbacks["markersize"](self.line_params)
        kwargs["color"] = self.callbacks["color"](self.line_params)

        delete_list = []
        for key, value in kwargs.items():
            if value is None:
                delete_list.append(key)

        for key in delete_list:
            kwargs.pop(key)

        # r1 = list(map(lambda x: x[0]-x[1], zip(returnavg, returnstd)))
        # r2 = list(map(lambda x: x[0]+x[1], zip(returnavg, returnstd)))

        logging.info(kwargs)

        # if smooth:
        #     y_smoothed = gaussian_filter1d(y, sigma=1)
        #     dif = np.absolute(y_smoothed - y)
        #     r1 = y_smoothed - dif
        #     r2 = y_smoothed + dif
        #     ax.fill_between(x, r1, r2, color=kwargs["color"], alpha=0.2)
        #     ax.plot(x, y_smoothed, markerfacecolor='none', **kwargs)

        if "smooth" in self.line_params and self.line_params["smooth"] > 0:
            y_smoothed = gaussian_filter1d(y, sigma=self.line_params["smooth"])
            # dif = np.absolute(y_smoothed - y)
            # r1 = y_smoothed - dif
            # r2 = y_smoothed + dif
            # ax.fill_between(x, r1, r2, color=kwargs["color"], alpha=0.2)
            new_kwargs = copy.deepcopy(kwargs)
            new_kwargs["alpha"] = 0.1
            new_kwargs["label"] = None
            ax.plot(x, y, markerfacecolor='none', **new_kwargs)
            ax.plot(x, y_smoothed, markerfacecolor='none', **kwargs)
        else:
            ax.plot(x, y, markerfacecolor='none', **kwargs)



        # ax.plot(x, y, label=label, marker=marker,
        #         linewidth=linewidth, linestyle=linestyle,
        #         markersize=markersize,
        #         markerfacecolor='none', color=color, dashes=dashes)


def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)






def heatmap(data, col_ticks_to_show, row_ticks_to_show,  
            col_labels, row_labels,
            ax=None,
            annot=False,
            fmt='.2f',
            fontsize=15,
            cmap="viridis",
            vmin=None, vmax=None,
            cbar_kws={}, cbarlabel="",
            **kwargs):
    """
    cmap: viridis, YlGnBu

    """

    if "linewidths" in kwargs:
        linewidths = kwargs["linewidths"]
    else:
        linewidths = .2

    if "gap_line" in kwargs and kwargs["gap_line"] is False:
        sns.heatmap(data, ax=ax, fmt=fmt,
                    xticklabels=col_labels, yticklabels=row_labels,
                    cmap=cmap, annot=annot, vmin=vmin, vmax=vmax,
                    cbar_kws=cbar_kws)
    else:
        sns.heatmap(data, ax=ax, fmt=fmt, linewidths=linewidths,
                    xticklabels=col_labels, yticklabels=row_labels,
                    cmap=cmap, annot=annot, vmin=vmin, vmax=vmax,
                    cbar_kws=cbar_kws)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # update_fontsize(cbar.ax, fontsize)
    # cbar.ax.set_fontsize(fontsize)


    # ax.set_xticks(col_ticks_to_show)
    # ax.set_yticks(row_ticks_to_show)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                labeltop=False, labelbottom=True)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=0)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    # ax.grid(which="minor", color="r", linestyle='-', linewidth=0)
    # ax.tick_params(which="minor", bottom=False, left=False)
    # ax.set_ylabel(y_label)
    # ax.set_xlabel(x_label)
    update_fontsize(ax, fontsize=fontsize)

    return ax




def mpl_heatmap(data, row_ticks_to_show, col_ticks_to_show, 
            row_labels, col_labels, ax=None, fontsize=15,
            cbar_kw={}, cbarlabel="", x_label="", y_label="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    update_fontsize(cbar.ax, fontsize)
    # cbar.ax.set_fontsize(fontsize)

    # We want to show all ticks...
    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticks(col_ticks_to_show)
    ax.set_yticks(row_ticks_to_show)
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.grid(which="minor", color="r", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    update_fontsize(ax, fontsize=fontsize)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    We're not using this for our graphs
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts




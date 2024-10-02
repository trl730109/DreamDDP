from __future__ import print_function
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




if __name__ == '__main__':

    CIFAR10_RES18 = "CIFAR10_RES18"
    # build_run("hpml-hkbu/DDP-Train/ddlvt4ws", CIFAR10_RES18,
    #         {"": ""}, "sgd-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001")

    # build_run("hpml-hkbu/DDP-Train/4-worker", CIFAR10_RES18,
    # {"": ""}, "Resnet-18")
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
    # build_run("hpml-hkbu/DDP-Train/jrvt4f5h", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/3f8q5hmh", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/bjupi4ih", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/fzjevwvk", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/ooewfl95", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet18-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")
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
    # build_run("hpml-hkbu/DDP-Train/vlxjmtrc", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/p8rbp5mb", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/z8rij6nd", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/nif0tprr", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/z59drayy", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw4-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")

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




    # # build_run("hpml-hkbu/DDP-Train/32-worker", CIFAR10_RES18,
    # # {"": ""}, "Resnet-50")
    # build_run("hpml-hkbu/DDP-Train/1ijmvjh1", CIFAR10_RES18,
    # {"": ""}, "sgd-noiFalse-resnet50-nw32-SGD-LG20-lr0.1-bs128-")
    # build_run("hpml-hkbu/DDP-Train/kjr0zuig", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001")
    # build_run("hpml-hkbu/DDP-Train/945hp3wx", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001")
    # build_run("hpml-hkbu/DDP-Train/9v7ch8vu", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01")
    # build_run("hpml-hkbu/DDP-Train/qtaycdbw", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1")
    # build_run("hpml-hkbu/DDP-Train/6pal1kpl", CIFAR10_RES18,
    # {"": ""}, "sgd-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0")

    # build_run("hpml-hkbu/DDP-Train/alvxajcd", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/3zn8vl8y", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/hromcduy", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/7fb6n0w7", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/okha0hbv", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP5")
    # build_run("hpml-hkbu/DDP-Train/t4k8edy2", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/0j0fknwt", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/ud20b6zx", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/ugxlaz6z", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/o7tlqijx", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP10")
    # build_run("hpml-hkbu/DDP-Train/eerecwm0", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/e5r3b4k7", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/l5m9rqyc", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/ye3a6y1j", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP50")
    # build_run("hpml-hkbu/DDP-Train/uqn4k4e7", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP50")
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
    # # build_run("hpml-hkbu/DDP-Train/base", CIFAR10_RES18,
    # # {"": ""}, "detect")
    # build_run("hpml-hkbu/DDP-Train/pikfppnq", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.0001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/car3dlwu", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.001-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/pcjhol47", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.01-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/cvnjlh3t", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd0.1-SyncP100")
    # build_run("hpml-hkbu/DDP-Train/hj1doccl", CIFAR10_RES18,
    # {"": ""}, "sgd_with_sync-noiTrue-resnet50-nw32-SGD-LG20-lr0.1-bs128-nstd1.0-SyncP100")




    build_run("hpml-hkbu/DDP-Train/s7pehfbb", CIFAR10_RES18,
    {"": ""}, "sgd-noiFalse-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.01-SyncP5")

    build_run("hpml-hkbu/DDP-Train/yrjrd26u", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/138xfu4s", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/8bk5ni75", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.01-SyncP5")
    build_run("hpml-hkbu/DDP-Train/zark4z34", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.1-SyncP5")
    build_run("hpml-hkbu/DDP-Train/lgo7mnkc", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd1.0-SyncP5")

    # DetectBase
    build_run("hpml-hkbu/DDP-Train/zdpdzw77", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/e62vdsra", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/n3nh8428", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.01-SyncP5")
    build_run("hpml-hkbu/DDP-Train/rf0c6lvk", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.1-SyncP5")
    build_run("hpml-hkbu/DDP-Train/vb9u3amd", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd1.0-SyncP5")


    build_run("hpml-hkbu/DDP-Train/sv4glz1f", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/v85d0g46", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/z116b1u7", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.01-SyncP5")
    build_run("hpml-hkbu/DDP-Train/pp25ids8", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.1-SyncP5")
    build_run("hpml-hkbu/DDP-Train/1iveiku4", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd1.0-SyncP5")

    # Fix-sync-freq
    build_run("hpml-hkbu/DDP-Train/bu31npq5", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP10")
    build_run("hpml-hkbu/DDP-Train/0ncgtgey", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP10")

    build_run("hpml-hkbu/DDP-Train/4e0jp3gk", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP50")
    build_run("hpml-hkbu/DDP-Train/uqxn5bj0", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP50")


    build_run("hpml-hkbu/DDP-Train/lpsr86wq", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.0001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/jga3m1n8", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-lora-nw4-Adam-LG20-lr0.001-bs8-nstd0.001-SyncP5")








    build_run("hpml-hkbu/DDP-Train/7gtxye0z", CIFAR10_RES18,
    {"": ""}, "sgd-noiFalse-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.01")

    build_run("hpml-hkbu/DDP-Train/ie2wiq1p", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.0001")
    build_run("hpml-hkbu/DDP-Train/3srxbtig", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.001")
    build_run("hpml-hkbu/DDP-Train/h5nyg5my", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.01")
    build_run("hpml-hkbu/DDP-Train/l7n20qz1", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.1")


    build_run("hpml-hkbu/DDP-Train/yqod643r", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.0001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/f4s02bhe", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/e0advrvc", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.01-SyncP5")
    build_run("hpml-hkbu/DDP-Train/qxhxj4zu", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-llama2-7B-lora-nw4-Adam-LG20-lr0.0001-bs2-nstd0.1-SyncP5")










    build_run("hpml-hkbu/DDP-Train/9ox70zlh", CIFAR10_RES18,
    {"": ""}, "sgd-noifalse-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-1Nodes")

    # Noised SGD
    build_run("hpml-hkbu/DDP-Train/swkgf28w", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001")
    build_run("hpml-hkbu/DDP-Train/1c6h68fw", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.001")
    build_run("hpml-hkbu/DDP-Train/6vu5pv6t", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.01")
    build_run("hpml-hkbu/DDP-Train/rcgotgsv", CIFAR10_RES18,
    {"": ""}, "sgd-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.1")

    # SGD with Sync detect base
    build_run("hpml-hkbu/DDP-Train/f6kkx5zd", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001-SyncP51")
    build_run("hpml-hkbu/DDP-Train/wcldqi3y", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.001-SyncP51")
    build_run("hpml-hkbu/DDP-Train/2ft0pq1m", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.01-SyncP51")
    build_run("hpml-hkbu/DDP-Train/6k5a1k7i", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.1-SyncP51")

    # SGD with Sync all detect base
    build_run("hpml-hkbu/DDP-Train/92aizpbc", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001-SyncP5-detech")
    build_run("hpml-hkbu/DDP-Train/ooh6ai4f", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.001-SyncP5-detech")
    build_run("hpml-hkbu/DDP-Train/kbvwv80s", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.01-SyncP5-detech")
    build_run("hpml-hkbu/DDP-Train/fbiwcuq3", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.1-SyncP5-detech")

    # SGD with Sync all fix H=5,10,50
    build_run("hpml-hkbu/DDP-Train/48w1ugue", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/qlco3m1c", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001-SyncP10")
    build_run("hpml-hkbu/DDP-Train/p5y6ihwc", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.0001-SyncP50")

    build_run("hpml-hkbu/DDP-Train/rncysng4", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.001-SyncP5")
    build_run("hpml-hkbu/DDP-Train/figyfj20", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.001-SyncP10")
    build_run("hpml-hkbu/DDP-Train/n9dk76ok", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.001-SyncP50")

    build_run("hpml-hkbu/DDP-Train/fzeyhs5k", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.01-SyncP5")
    build_run("hpml-hkbu/DDP-Train/ef0xkdat", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.01-SyncP10")
    build_run("hpml-hkbu/DDP-Train/1nvqi2ab", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.01-SyncP50")

    build_run("hpml-hkbu/DDP-Train/92gowe9m", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.1-SyncP5")
    build_run("hpml-hkbu/DDP-Train/bc6xn7wj", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.1-SyncP10")
    build_run("hpml-hkbu/DDP-Train/hieek5s3", CIFAR10_RES18,
    {"": ""}, "sgd_with_sync_all-noiTrue-tfix-gpt2-full-nw8-Adam-LG20-lr1e-3-bs8-nstd0.1-SyncP50")






































































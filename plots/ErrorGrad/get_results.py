from __future__ import print_function
import os
import sys
import logging
from turtle import down

import pandas as pd
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))


from utils.wandb_util import get_project_runs_from_wandb, download_run
from utils.experiment_util import (
    ExpRun, ExpBook,
    get_alias,
    get_alias_from_list,
    generate_alias_list,
    find_one_uid,
    generate_config_key,
    combine_config,
    get_summary_name
)

from utils.common import *



FedAvg_name = 'FedAvg'


renset18_torch = "renset18_torch"
renset18 = "renset18"

albert_base_v2 = "albert-base-v2"

cifar10 = "cifar10"
cifar100 = "cifar100"
femnist = "femnist"
stackoverflow = "stackoverflow_nwp"
reddit = "reddit"
reddit_blog = "blog"


fedml = "FedML"
fedscale = "FedScale"
flower = "Flower"


model_list = []


con_client_list = [10, 100] # '32'
con_client_10 = 10
con_client_100 = 100

local_epoch_5 = 5
local_epoch_10 = 10
local_epoch_100 = 100

entity = "hpml-hkbu"
project = "FusionAI"
root_path = "D://1results//wandb_backup"

# all_df, runs_dict = get_project_runs_from_wandb(entity=entity, project=project,
#                             no_summary=True)
# exp_book = ExpBook(entity, project, root_path, all_df, runs_dict,
#                 filter_name=TEST_LOSS, sort_value_name=TEST_LOSS,
#                 sort=True, ascending=False)

api = wandb.Api()
# api = wandb.public.Api()



# same figure_name as key, alias of run as value.
alias_run_map = {}
all_figures = {}


TRAIN_CE_LOSS_EMA = "train/loss_cross_entropy_ema"
TRAIN_ACC = "train/classification_acc"
TRAIN_GEN_LOSS_EMA = "train/generation_loss_ema"
MPG0_LOCAL_ITER = "local_iter/MPG_0"

TOTAL_DIVER = "total_diversity"
TIME_PER_ITER = "time_per_iter"
TRAIN_ACC = "train_epoch_acc"
VAL_ACC = "val_acc"
TRAIN_LOSS = "train_epoch_loss"
EPOCHS = "epochs"
ITERS = "global_iters"

TIME_PER_ITER = "time_per_iter"



"""
    All config parameters should be same as config on wandb.
"""


def load_data(key, key2, history):
    # data1 = np.array(history[key][history[key].notnull()])
    # data2 = np.array(history[history[key].notnull()][key2])
    data1 = []
    data2 = []
    for row in history:
        if key in row:
            data1.append(row[key])
            data2.append(row[key2])
    data1 = np.array(data1)
    data2 = np.array(data2)
    print(f"{key}: {data1}, {key2}: {(data2)}")
    print(f"{key}: length: {len(data1)}, {key2}: length: {len(data2)}")
    return data1, data2


def load_multiple_data(key_list, history):
    # data1 = np.array(history[key][history[key].notnull()])
    # data2 = np.array(history[history[key].notnull()][key2])
    datas = dict(list([(key, []) for key in key_list]))
    key0 = key_list[0]

    for row in history:
        if key0 in row:
            for key in key_list:
                datas[key].append(row[key])

    for key in key_list:
        datas[key] = np.array(datas[key])
        print(f"{key}: length: {len(datas[key])}")
    return datas


def filter_ourliers(data, cutlen=5, times_std=2):
    data = np.array(data)
    data = data[cutlen:-cutlen]
    mean = data.mean()
    std = data.std()
    print(f"meam: {mean}, std: {std}, len: {len(data)}")
    data = data[(data < mean + times_std*std) & (data > mean - times_std*std)]
    mean = data.mean()
    std = data.std()
    print(f"new meam: {mean}, new std: {std}, len: {len(data)}")
    return data, mean, std



def load_datas(key, key2, alias_list, verbose=False):
    # alias_list = all_figures[to_figure_name]
    data_dict = {}
    data_dict2 = {}
    std_dict = {}
    for alias in alias_list:
        history = get_history(alias)
        # logging.info(history)
        data1, data2 = load_data(key, key2, history)
        if verbose:
            logging.info(data1)
            logging.info(data2)
        data_dict[alias] = data1
        data_dict2[alias] = data2
    return data_dict, data_dict2



def load_summarys(keys, alias_list, verbose=False):
    # alias_list = all_figures[to_figure_name]
    data_dict = {}
    for alias in alias_list:
        summary = get_summary(alias)
        logging.info(summary)
        data_dict[alias] = {}
        for key in keys:
            data_dict[alias][key] = summary[key]
        if verbose:
            logging.info(f"{alias}: has {key} as {summary[key]}")
    return data_dict



def load_multiple_datas(key_list, alias_list, verbose=False):
    # alias_list = all_figures[to_figure_name]
    data_dict = {}
    std_dict = {}
    for alias in alias_list:
        history = get_history(alias)
        # logging.info(history)
        datas = load_multiple_data(key_list, history)
        if verbose:
            logging.info(datas)
        data_dict[alias] = datas
    return data_dict







def filter_none(data_dict, data_dict2):
    for alias in data_dict.keys():
        # data1s = data_dict[alias]
        # data2s = data_dict2[alias]
        # data1s_without_none = data1s[data1s != None]
        # data1s_cleaned = data1s_without_none[~np.isnan(data1s_without_none)]
        # data2s_without_none = data1s[data1s != None]
        # data2s_cleaned = data2s_without_none[~np.isnan(data2s_without_none)]
        # data1_not_none = data_dict[alias] != None
        # data_dict[alias] = data_dict[alias][data1_not_none]
        # data_dict2[alias] = data_dict2[alias][data1_not_none]

        data_dict[alias] = np.array(data_dict[alias], dtype=float)
        data_dict2[alias] = np.array(data_dict2[alias], dtype=float)

        data1_not_nan = ~np.isnan(data_dict[alias])
        data_dict[alias] = data_dict[alias][data1_not_nan]
        data_dict2[alias] = data_dict2[alias][data1_not_nan]

        # data2_not_none = data_dict[alias] != None
        # data_dict[alias] = data_dict[alias][data2_not_none]
        # data_dict2[alias] = data_dict2[alias][data2_not_none]

        data2_not_nan = ~np.isnan(data_dict[alias])
        data_dict[alias] = data_dict[alias][data2_not_nan]
        data_dict2[alias] = data_dict2[alias][data2_not_nan]
        # for i, data in enumerate(zip(data1s, data2s)):



def filter_multiple_none(datas_dict):
    for alias in datas_dict.keys():
        for metric in datas_dict[alias].keys():
            datas_dict[alias][metric] = np.array(datas_dict[alias][metric], dtype=float)

        for metric_i in datas_dict[alias].keys():
            data1_not_nan = ~np.isnan(datas_dict[alias][metric_i])
            for metric_j in datas_dict[alias].keys():
                # if metric_i == metric_j:
                #     pass
                # else:
                datas_dict[alias][metric_j] = datas_dict[alias][metric_j][data1_not_nan]




def load_data_figure(key, key2, alias_list, filter=True):
    # alias_list = all_figures[to_figure_name]
    data_dict = {}
    mean_dict = {}
    std_dict = {}
    for alias in alias_list:
        history = get_history(alias)
        data1, data2 = load_data(key, key2, history)
        if filter:
            data1, mean, std = filter_ourliers(data1, cutlen=5, times_std=2)
        data_dict[alias] = data1
        mean_dict[alias] = mean
        std_dict[alias] = std
    return data_dict, mean_dict, std_dict


def build_run(uid, to_figure_name, help_params, alias, config={}, *args):
    if alias is None:
        # alias = get_alias_from_list(*args)
        alias = get_alias(config)
    print(f"alias: {alias}")
    config = config
    alias_run_map[alias] = {
        "config": config,
        "help_params": help_params,
        "uid": uid,
        "run": download_run(api, root_path=root_path, uid=uid),
        "downloaded": False,
        "history": None
    }

    if to_figure_name not in all_figures:
        all_figures[to_figure_name] = []
        all_figures[to_figure_name].append(alias)
    else:
        if alias not in all_figures[to_figure_name]:
            all_figures[to_figure_name].append(alias)


def get_run(alias):
    return alias_run_map[alias]["run"]


def get_summary(alias):
    run = alias_run_map[alias]["run"]
    if not alias_run_map[alias]["downloaded"]:
        # history_scan = run.scan_history()
        # alias_run_map[alias]["history"] = pd.DataFrame.from_records([row for row in history_out])
        # alias_run_map[alias]["history"] = history_scan
        # alias_run_map[alias]["history"] = run.history
        alias_run_map[alias]["downloaded"] = True
    return run.summary


def get_history(alias):
    run = alias_run_map[alias]["run"]
    if not alias_run_map[alias]["downloaded"]:
        # history_scan = run.scan_history()
        # alias_run_map[alias]["history"] = pd.DataFrame.from_records([row for row in history_out])
        # alias_run_map[alias]["history"] = history_scan
        alias_run_map[alias]["history"] = run.history
        alias_run_map[alias]["downloaded"] = True
    return alias_run_map[alias]["history"]



if __name__ == '__main__':
    pass
































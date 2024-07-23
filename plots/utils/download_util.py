import logging
import numpy as np

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



def load_datas(key, key2, alias_list):
    # alias_list = all_figures[to_figure_name]
    data_dict = {}
    data_dict2 = {}
    std_dict = {}
    for alias in alias_list:
        history = get_history(alias)
        logging.info(history)
        data1, data2 = load_data(key, key2, history)
        data_dict[alias] = data1
        data_dict2[alias] = data2
    return data_dict, data_dict2



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


def build_run(api, all_figures, alias_run_map,
        uid, to_figure_name, help_params, alias, config={}, *args):
    if alias is None:
        # alias = get_alias_from_list(*args)
        alias = get_alias(config)
    print(f"alias: {alias}")
    config = config
    alias_run_map[alias] = {
        "config": config,
        "help_params": help_params,
        "uid": uid,
        "run": download_run(api, uid=uid),
        "downloaded": False,
        "history": None
    }

    if to_figure_name not in all_figures:
        all_figures[to_figure_name] = []
        all_figures[to_figure_name].append(alias)
    else:
        if alias not in all_figures[to_figure_name]:
            all_figures[to_figure_name].append(alias)


def get_run(alias_run_map, alias):
    return alias_run_map[alias]["run"]


def get_history(alias_run_map, alias):
    run = alias_run_map[alias]["run"]
    if not alias_run_map[alias]["downloaded"]:
        # history_scan = run.scan_history()
        # alias_run_map[alias]["history"] = pd.DataFrame.from_records([row for row in history_out])
        # alias_run_map[alias]["history"] = history_scan
        alias_run_map[alias]["history"] = run.history
        alias_run_map[alias]["downloaded"] = True
    return alias_run_map[alias]["history"]




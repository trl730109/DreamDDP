import logging
import os, sys
import copy
from pathlib import Path

import pandas as pd 
import wandb
import numpy as np

import pickle
import shutil

# from utils.meter import AverageMeter

wandb_api = wandb.Api()

def get_uid_path(entity, project, id, uid):

    if uid is not None:
        # new_uid = uid
        if "runs" not in uid:
            uid_strs = uid.split('/')
            uid_strs.append(copy.deepcopy(uid_strs[2]))
            uid_strs[2] = "runs"
            new_uid = '/'.join(uid_strs)
        else:
            new_uid = uid
        print(f"new_uid: {new_uid}")
    else:
        new_uid = f"{entity}/{project}/runs/{id}"
        # new_uid = f"{entity}/{project}/{id}"
    return new_uid


def pickle_save(data_obj, file_path):
    with open(file_path, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data_obj, f)


def pickle_load(file_path):
    if file_path.exists() and os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as f:
            loaded_obj = pickle.load(f)
    else:
        loaded_obj = {}
    return loaded_obj


# def wandb_log(prefix, sp_values, com_values, update_summary=False, wandb_summary_dict={}):
#     """
#         prefix + tags.values is the name of sp_values;
#         values should include information like:
#         {"Acc": 0.9, "Loss":}
#         com_values should include information like:
#         {"epoch": epoch, }
#     """
#     new_values = {}
#     for k, _ in sp_values.items():
#         # new_values[prefix+"/" + k] = sp_values[k]
#         new_key = prefix+"/" + k
#         new_values[new_key] = sp_values[k]
#         if update_summary:
#             if new_key not in wandb_summary_dict:
#                 wandb_summary_dict[new_key] = AverageMeter()
#             wandb_summary_dict[new_key].update(new_values[new_key], n=1)
#             summary = wandb_summary_dict[new_key].make_summary(new_key)
#             for key, valaue in summary.items():
#                 wandb.run.summary[key] = valaue

#     new_values.update(com_values)
#     wandb.log(new_values)


class ExpRun(object):
    # def __init__(self, entity, project, id, uid, config, run, 
    #             url=None, state=None, created_at=None, system_metrics=None,
    #             summary=None, file=None, generate_name=None):
    def __init__(self, uid, run):
        self.entity = copy.deepcopy(run.entity) 
        self.project = copy.deepcopy(run.project)
        self.id = copy.deepcopy(run.id)
        self.uid = copy.deepcopy(uid)
        # assert self.uid == run.entity + "/" + run.project + "/" + run.id
        self.config = copy.deepcopy(run.config)
        self.url = copy.deepcopy(run.url)
        self.state = copy.deepcopy(run.state) 
        self.created_at = copy.deepcopy(run.created_at)
        self.created_at_number_str = time_to_number_str(self.created_at)
        self.system_metrics = copy.deepcopy(run.system_metrics)
        try:
            self.summary = copy.deepcopy(dict(run.summary))
        except:
            pass

        wandb_hist = run.scan_history()
        list_hist = []
        # logging.info(f"len(wandb_hist): {len(wandb_hist)}")
        for i, row in enumerate(wandb_hist):
            list_hist.append(row)
        self.history = list_hist
        # self.file = run.file

    # def download_file(self, file_name, root_path, replace=True):
    #     # path = get_run_path(root_path)
    #     path = os.path.join(root_path, self.run_path)
    #     self.run.file(file_name).download(path, replace=replace)

    def get_history(self):
        return self.history


    def download_wandb_basic_files(self, root_path, replace=True):
        # path = get_run_path(root_path)
        path = os.path.join(root_path, self.run_path)
        # self.run.file("requirements.txt").download(path, replace=replace)
        # self.run.file("config.yaml").download(path, replace=replace)
        # self.run.file("wandb-metadata.json").download(path, replace=replace)
        # self.run.file("wandb-summary.json").download(path, replace=replace)

    # def get_file_from_wandb(self, file_name):
    #     return self.run.file(file_name)


    # def refresh_history(self, unsample=False):
    #     """
    #         When the points on wandb exceed 500, the run.history() will automatically unsample points.
    #         So may use scan_history()
    #     """
    #     if unsample:
    #         self.history = self.run.history()
    #     else:
    #         history_out = self.run.scan_history()
    #         # if isinstance(history_out, wandb.apis.public.HistoryScan):
    #         self.history = pd.DataFrame.from_records([row for row in history_out])

    #     return self.history





def delete_output_log(path=""):
    api = wandb.Api()
    runs = api.runs(path)
    for run in runs:
        log = run.file("output.log")
        if log and log.size > 0:
            print("Log: {}, size: {}, executing delete....".format(log, log.size))
            log.delete()
        else:
            print("Log: {}, pass....".format(log))



def download_run_new(entity=None, project=None, id=None, uid=None):
    if uid is not None:
        if "runs" not in uid:
            uid_strs = uid.split('/')
            uid_strs.append(copy.deepcopy(uid_strs[2]))
            uid_strs[2] = "runs"
            new_uid = '/'.join(uid_strs)
            run = wandb_api.run(new_uid)
        else:
            run = wandb_api.run(uid)
    else:
        run = wandb_api.run(f"{entity}/{project}/runs/{id}")
    return run



def download_run(api, root_path, entity=None, project=None, id=None, uid=None, cache=True):
    if uid is not None:
        # new_uid = uid
        if "runs" not in uid:
            uid_strs = uid.split('/')
            uid_strs.append(copy.deepcopy(uid_strs[2]))
            uid_strs[2] = "runs"
            new_uid = '/'.join(uid_strs)
        else:
            new_uid = uid
        print(f"new_uid: {new_uid}")
    else:
        new_uid = f"{entity}/{project}/runs/{id}"
        # new_uid = f"{entity}/{project}/{id}"

    # root_dir = Path("wandb_results")
    root_dir = Path(root_path)
    run_hist_path = root_dir / f"{new_uid}.pickle"
    if cache:
        print(f"new_uid: {new_uid}")
        if not root_dir.exists():
            # os.makedirs(root_dir)
            root_dir.mkdir(parents=True)
        if not run_hist_path.exists():
            if not run_hist_path.parent.exists():
                run_hist_path.parent.mkdir(parents=True)
            run = api.run(new_uid)
            exp_run = ExpRun(new_uid, run)
            # history = exp_run.history
            pickle_save(exp_run, run_hist_path)
        else:
            exp_run = pickle_load(run_hist_path)
    else:
        run = api.run(new_uid)
        exp_run = ExpRun(new_uid, run)
    return exp_run


def get_project_runs_from_wandb(entity, project, filters={}, order="-created_at", per_page=50,
                                no_summary=False):
    """
        path="", filters={}, order="-created_at", per_page=50
        run: A single run associated with an entity and project.
        Attributes:
            tags ([str]): a list of tags associated with the run
            url (str): the url of this run
            id (str): unique identifier for the run (defaults to eight characters)
            name (str): the name of the run
            state (str): one of: running, finished, crashed, aborted
            config (dict): a dict of hyperparameters associated with the run
            created_at (str): ISO timestamp when the run was started
            system_metrics (dict): the latest system metrics recorded for the run
            summary (dict): A mutable dict-like property that holds the current summary.
                        Calling update will persist any changes.
            project (str): the project associated with the run
            entity (str): the name of the entity associated with the run
            user (str): the name of the user who created the run
            path (str): Unique identifier [entity]/[project]/[run_id]
            notes (str): Notes about the run
            read_only (boolean): Whether the run is editable
            history_keys (str): Keys of the history metrics that have been logged
                with `wandb.log({key: value})`
    """

    api = wandb.Api()
    # Project is specified by <entity/project-name>
    # usage: path="", filters={}, order="-created_at", per_page=50
    path = entity + "/" + project
    runs = api.runs(path, filters, order, per_page)
    summary_list = []
    config_list = []
    name_list = []
    id_list = []
    project_list = []
    uid_list = []
    state_list = []
    url_list = []

    username_list = []
    entity_list = []
    created_at_list = []

    runs_dict = {}
    for run in runs: 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        if no_summary:
            summary_list.append([])
        else:
            summary_list.append(run.summary._json_dict) 
        id_list.append(run.id)
        project_list.append(run.project)

        uid = run.entity + '/' + run.project + '/' + run.id
        uid_list.append(uid)
        runs_dict[uid] = run
        state_list.append(run.state)
        url_list.append(run.url)

        username_list.append(run._attrs['user']['username'])
        entity_list.append(run.entity)
        created_at_list.append(run.created_at)
        # run.config is the input metrics.
        # We remove special values that start with _.
        # config = {k:v for k,v in run.config.items() if not k.startswith('_')}
        config = {}
        for k, v in run.config.items():
            if not k.startswith('_'):
                if type(v) == list:
                    config[k] = str(v)
                else:
                    config[k] = v
        config_list.append(config) 

        # run.name is the name of the run.
        name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list})
    id_df = pd.DataFrame({'id': id_list})
    project_df = pd.DataFrame({'project': project_list})
    uid_df = pd.DataFrame({'uid': uid_list})
    state_df = pd.DataFrame({'state': state_list})
    url_df = pd.DataFrame({'url': url_list})

    username_df = pd.DataFrame({'Username': username_list})
    entity_df = pd.DataFrame({'entity': entity_list})
    created_at_df = pd.DataFrame({'created_at': created_at_list})

    all_df = pd.concat([name_df, config_df, summary_df, id_df, project_df, uid_df,
                        state_df, url_df, username_df, entity_df, created_at_df], axis=1)
    # all_df.to_csv("project.csv")
    return all_df, runs_dict


def load_summary_df(root_path, entity, project, reload=False):
    # summary_df_path = self.root_path / "summary.csv"
    summary_df_path = Path(root_path) / entity / project / "summary_df.csv"
    print(f"summary_df_path:{summary_df_path}")
    if not summary_df_path.exists() or reload:
        if not summary_df_path.parent.exists():
            summary_df_path.parent.mkdir(parents=True)
        all_df, runs_dict = get_project_runs_from_wandb(entity=entity, project=project)
        all_df.to_csv(summary_df_path)
    else:
        # usecols = ['type', 'title', 'director', 'date_added', 'rating']
        all_df = pd.read_csv(summary_df_path)
    return all_df



def get_project_run(entity, project, filters={}, order="-created_at", per_page=50,
                                no_summary=False):
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    # usage: path="", filters={}, order="-created_at", per_page=50
    path = entity + "/" + project
    runs = api.runs(path, filters, order, per_page)
    return runs




def time_to_number_str(time):
    day = time.split('T')[0]
    time = time.split('T')[1]
    day = day.replace("-", "")
    time = time.replace(":", "")
    new_time = day + "_" + time
    return new_time

def number_str_to_time(number_str):
    day = number_str.split('_')[0]
    time = number_str.split('_')[1]
    day = day[0:3] + "-" + day[4:6] + "-" + day[6:]
    time = time[0:2] + ":" + time[2:4] + ":" + time[4:]
    new_time = day + "T" + time
    return new_time




def load_data(key, key2, history):
    # data1 = np.array(history[key][history[key].notnull()])
    # data2 = np.array(history[history[key].notnull()][key2])
    data1 = []
    data2 = []
    for row in history:
        if key in row and key2 in row:
            if row[key] is not None and row[key2] is not None:
                data1.append(row[key])
                data2.append(row[key2])
    data1 = np.array(data1)
    data2 = np.array(data2)
    # print(f"{key}: {data1}, {key2}: {(data2)}")
    # print(f"{key}: length: {len(data1)}, {key2}: length: {len(data2)}")
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


def load_datas(key, key2, run_dict, filter=False):
    # alias_list = all_figures[to_figure_name]
    data_dict = {}
    data_dict2 = {}
    std_dict = {}
    for alias, run in run_dict.items():
        history = run.get_history()
        # logging.info(history)
        data1, data2 = load_data(key, key2, history)
        if filter:
            data1, mean, std = filter_ourliers(data1, cutlen=5, times_std=2)
        data_dict[alias] = data1
        data_dict2[alias] = data2
    return data_dict, data_dict2









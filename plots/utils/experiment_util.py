import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
import copy

import pandas as pd
import numpy as np
import pickle

from .wandb_util import (
    # get_project_path,
    # get_run_folder_name,
    # get_run_path,
    time_to_number_str,
    number_str_to_time,
    get_uid_path,
    wandb_api,
    ExpRun,
    download_run,
    get_project_runs_from_wandb,
)

def get_alias(config):
    return "-".join([str(key)+"="+str(config[key]) for key in config.keys()])


def get_alias_from_list(*args):
    return "-".join([str(arg) for arg in args])





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


def generate_alias_list(
    alias_run_map, exp_book,
    basic_config={}, fixed_config={}, varying_config_list=[],
    fixed_help_params={}, varing_help_params_list=[],
    uid_list=[], find_uid_func=None, find_uid_kargs=None):
    """
        use alias_run_map to record,
        All config parameters should be same as config on wandb.
    """

    config_list = []
    help_params_list = []
    alias_list = []

    new_basic_config = combine_config(basic_config, fixed_config)

    for varying_config in varying_config_list:
        config_list.append(combine_config(new_basic_config, varying_config))

    basic_help_params = {}
    basic_help_params = combine_config(basic_help_params, fixed_help_params)

    if len(varing_help_params_list) > 0:
        for varing_help_params in varing_help_params_list:
            help_params_list.append(combine_config(basic_help_params, varing_help_params))

    run_dict = {}
    for i, config in enumerate(config_list):
        if len(varing_help_params_list) > 0:
            help_params = help_params_list[i]
        else:
            help_params = basic_help_params

        if len(uid_list) > 0:
            uid = uid_list[i]
        else:
            uid = None

        alias = get_alias(config)
        alias_list.append(alias)
        alias_run_map[alias] = {
            "config": config,
            "help_params": help_params,
            "uid": uid,
            "find_uid_func": find_uid_func,
            "find_uid_kargs": find_uid_kargs,
            "downloaded": False,
            "history": None
        }
        config_key, uid, exp_run = exp_book.add_config(
            config=config, alias=alias,
            help_params=help_params, uid=uid,
            find_uid_func=find_uid_func)
        run_dict[alias] = exp_run

    return alias_list, run_dict



def modify_dict(dict, key_func=None, value_func=None):
    new_dict = {}
    for key, value in dict.items():
        if key_func is not None:
            new_dict[key_func(key)] = value_func(value)
        else:
            new_dict[key] = value_func(value)
    return new_dict


def modify_list(list, value_func=None):
    new_list = []
    for value in list:
        new_list.append(value_func(value))
    return new_list


def combine_config(basic_config, update_config):
    new_config = deepcopy(basic_config)
    for k in list(update_config.keys()):
        new_config[k] = update_config[k]
    return new_config

def extend_dicts_from_list(old_dict_list, name, value_list):
    new_dict_list = []

    for old_dict in old_dict_list:
        if value_list is None or len(value_list) == 0:
            new_dict = deepcopy(old_dict)
            new_dict[name] = None
            new_dict_list.append(new_dict)
        else:
            for value in value_list:
                new_dict = deepcopy(old_dict)
                new_dict[name] = value
                new_dict_list.append(new_dict)
    return new_dict_list


def build_dicts(dict_of_list_attributes, basic_dict_list):
    new_dict_list = basic_dict_list
    for name, value_list in dict_of_list_attributes.items():
        new_dict_list = extend_dicts_from_list(
            old_dict_list=new_dict_list,
            name=name,
            value_list=value_list
        )
    return new_dict_list



def postfix_process(
    client_index=None,
    server_index=None,
    if_global=False
):
    postfix = ""
    if client_index is not None and \
        (server_index is None and if_global is False):
        postfix += '/client' + str(client_index)
    elif server_index is not None and \
        (client_index is None and if_global is False):
        postfix += '/server' + str(server_index)
    elif if_global and \
        (server_index is None and client_index is None):
        postfix += '/' + "global"
    elif if_global is False and \
        (server_index is None and client_index is None):
        pass
    else:
        raise RuntimeError

    return postfix



def return_name_in_dict(name, dict, default):
    if name in dict:
        return dict[name]
    else:
        return default


def get_summary_name(
    **summary_name_dict
):
    root = summary_name_dict["line_mode"] + '/' + summary_name_dict["thing"]

    if "LP" in summary_name_dict and summary_name_dict["LP"] is not None:
        root += '/LP' + str(summary_name_dict["LP"]) 

    if "layer" in summary_name_dict and summary_name_dict["layer"] is not None:
        root += '/' + summary_name_dict["layer"]

    root += postfix_process(
        client_index=return_name_in_dict("client_index", summary_name_dict, None),
        server_index=return_name_in_dict("server_index", summary_name_dict, None),
        if_global=return_name_in_dict("if_global", summary_name_dict, False),
    )

    return root




def strip_summary_name(total_summary_name):
    stripped_summary_name_list = total_summary_name.split("/")
    return stripped_summary_name_list[1]


def get_metric_params(line_mode, thing, layers=None, LP_list=None,
        client_list=None, server_list=None, if_global=False
    ):
    basic_dict = {
        "line_mode": line_mode,
        "thing": thing
    }

    new_dict_list = [basic_dict]

    if layers is not None:
        new_dict_list = build_dicts(
            dict_of_list_attributes={"layer": layers},
            basic_dict_list=new_dict_list
        )

    if LP_list is not None:
        new_dict_list = build_dicts(
            dict_of_list_attributes={"LP": LP_list},
            basic_dict_list=new_dict_list
        )

    output_dict_list = []
    if client_list is not None and len(client_list) > 0:
        output_dict_list += build_dicts(
            dict_of_list_attributes={"client_index": client_list},
            basic_dict_list=new_dict_list
        )


    if server_list is not None and len(server_list) > 0:
        output_dict_list += build_dicts(
            dict_of_list_attributes={"server_index": server_list},
            basic_dict_list=new_dict_list
        )

    if if_global:
        output_dict_list += build_dicts(
            dict_of_list_attributes={"if_global": [if_global]},
            basic_dict_list=new_dict_list
        )

    if client_list is None and server_list is None and not if_global:
        output_dict_list = new_dict_list

    return output_dict_list



def get_metric_things(line_mode, thing, layers=None, LP_list=None,
        client_list=None, server_list=None, if_global=False
    ):

    output_dict_list = get_metric_params(line_mode, thing,
        layers=layers, LP_list=LP_list,
        client_list=client_list, server_list=server_list,
        if_global=if_global)

    things_list = []
    for output_dict in output_dict_list:
        things_list.append(get_summary_name(**output_dict))

    return things_list




def get_multi_metric_things(alias_list, 
    line_mode, all_thing_list,
):
    new_all_thing_list = []
    for thing in all_thing_list:
        things_list = get_metric_things(line_mode, thing)
        new_all_thing_list += things_list

    logging.info(f"things_list: {things_list}")

    alias_metric_things_dict = {}
    for alias in alias_list:
        alias_metric_things_dict[alias] = new_all_thing_list
    return alias_metric_things_dict



def get_same_alias_metric_things(alias_list, 
    line_mode, thing, layers=None, client_list=None, if_global=False
):
    things_list = get_metric_things(
        line_mode, thing, layers=None, client_list=None, if_global=if_global)

    logging.info(f"things_list: {things_list}")

    alias_metric_things_dict = {}
    for alias in alias_list:
        alias_metric_things_dict[alias] = things_list
    return alias_metric_things_dict


def generate_config_key(config):
    """
    This method is used to generate the name of one experiment record.
    Note that here the name is not unique, because we may want to change codes 
    between two experiments, or there may be only a part of arguments added into the name.
    Also, the name will be used as the name of the wandb run.
    """
    return "-".join([str(key)+"="+str(config[key]) for key in config.keys()])


def get_legend_name(prefix_and_names: list, map_seg='', seg='-'):
    """
    Each item in prefix_and_names shoule be (prefix, name)
    """
    legend_name = ""
    for i, item in enumerate(prefix_and_names):
        if i == prefix_and_names:
            legend_name += str(item[0]) + map_seg + str(item[1]) + seg
        else:
            legend_name += str(item[0]) + map_seg + str(item[1])


def check_get_run(exp_book, out_df, sort_value_name, ascending):
    try:
        if len(out_df) > 0:
            # logging.info("WARNING, The number of results found out is more than one.\
            #     There maybe duplicate experiments!!!!!!!")
            # out_df = out_df.sort_values(sort_value_name, ascending=ascending)
            # uid = out_df.iloc[0]["uid"]
            # exp_run = exp_book.get_run(uid)
            # detail_config = exp_run.config
            # logging.info("The filtered detailed config is {}".format(
            #     detail_config
            #     ))
            # else:
            #     uid = out_df.iloc[0]["uid"]
            #     exp_run = exp_book.get_run(uid)
            return True
    except:
        return False



def find_one_uid(exp_book, config=None, help_params=None, filter_name=None, sort_value_name=None,
                sort=True, ascending=False):
    out_df = exp_book.all_df
    # filter according to config
    for key in config.keys():
        if key == "b_created_at":
            out_df = out_df.loc[out_df["created_at"] > config["b_created_at"]]
            continue
        elif key == "s_created_at":
            out_df = out_df.loc[out_df["created_at"] < config["s_created_at"]]
            continue
        else:
            pass

        if key == "fed_noise_dataset_batch_size":
            if config[key] is None or config[key] == 128:
                # if config[key] is None:
                out_df_128 = out_df.loc[out_df[key] == 128]
                out_df_None = out_df.loc[out_df[key].isnull()]
                out_df = pd.concat([out_df_128, out_df_None])
                print(f"len(out_df_128): {len(out_df_128)} and len(out_df_None): {len(out_df_None)}"+\
                    f"len(out_df): {len(out_df)}")
                # continue
                continue
            else:
                pass


        if key == "model_out_feature_layer":
            if config[key] is None or config[key] == "last":
                # if config[key] is None:
                out_df_value = out_df.loc[out_df[key] == "last"]
                out_df_None = out_df.loc[out_df[key].isnull()]
                out_df = pd.concat([out_df_value, out_df_None])
                print(f"len(out_df_value): {len(out_df_value)} and len(out_df_None): {len(out_df_None)}"+\
                    f"len(out_df): {len(out_df)}")
                # continue
                continue


        if key == "fed_noise_feat_align_inter_cls_weight":
            if config[key] is None or config[key] == 1.0:
                # if config[key] is None:
                out_df_value = out_df.loc[out_df[key] == 1.0]
                out_df_None = out_df.loc[out_df[key].isnull()]
                out_df = pd.concat([out_df_value, out_df_None])
                print(f"len(out_df_value): {len(out_df_value)} and len(out_df_None): {len(out_df_None)}"+\
                    f"len(out_df): {len(out_df)}")
                # continue
                continue


        if key == "fed_noise_noise_contrastive":
            if config[key]:
                continue


        if config[key] is None:
            out_df = out_df.loc[out_df[key].isnull()]
            # continue
        else:
            pass

        # if key == "fed_noise_dataset_list":
        #     out_df = out_df.loc[config[key] in out_df[key]] 
        #     continue

        if config[key] == 'no':
            out_df_no = out_df.loc[out_df[key] == config[key]]
            out_df_None = out_df.loc[out_df[key].isnull()]
            out_df = pd.concat([out_df_no, out_df_None])
            continue

        if not config[key]:
            out_df_no = out_df.loc[out_df[key] == config[key]]
            out_df_None = out_df.loc[out_df[key].isnull()]
            out_df = pd.concat([out_df_no, out_df_None])
            continue

        # if config[key] == 'no':
        #     test_out_df = out_df.loc[out_df[key] == config[key]]
        #     find_result = check_get_run(exp_book, test_out_df, sort_value_name, ascending)
        #     if find_result:
        #         # logging.info("Find config: {} with key {} = no".format(config, key))
        #         out_df = test_out_df
        #         continue
        #     else:
        #         # logging.info("Find config: {} with key {} = None".format(config, key))
        #         out_df = out_df.loc[out_df[key].isnull()]
        #         continue

        out_df = out_df.loc[out_df[key] == config[key]]



    # filter according to filter_name
    # if len(out_df) > 0:
    logging.info("len(out_df) is {} ".format(len(out_df)))
    if filter_name is None:
        test_df = out_df
    else:
        test_df = out_df.loc[out_df[filter_name].notnull()]
    if len(test_df) > 0:
        out_df = test_df
    else:
        logging.info("len(out_df) is {} there is no \"{}\" summary item".format(
            len(out_df), filter_name))


    # for key in config.keys():
    #     adf = out_df.loc[out_df[key] == config[key]]
    # # filter according to filter_name
    # adf = adf.loc[adf['Test/Acc'].notnull()]
    # sort runs according to sort_value_name
    if sort:
        try:
            if len(out_df) > 1:
                logging.info("WARNING, The number of results found out is more than one.\
                    There maybe duplicate experiments!!!!!!!")
                out_df = out_df.sort_values(sort_value_name, ascending=ascending)
                uid = out_df.iloc[0]["uid"]
                exp_run = exp_book.get_run(uid)
                detail_config = exp_run.config
                logging.info("The filtered exp uid is {}".format(
                    uid
                ))
                for i in range(len(out_df) -1):
                    uid_other = out_df.iloc[i+1]["uid"]
                    logging.info(" ==========   other uid is:  {},".format(
                        uid_other
                    ))
            else:
                uid = out_df.iloc[0]["uid"]
                exp_run = exp_book.get_run(uid)
            try:
                logging.info("config: {}, find uid successfully, is: {}, the {} is {}".format(
                    config, uid, filter_name, exp_run.summary[filter_name]))
            except:
                logging.info("config: {}, find uid successfully, is: {}, but {} is not in summary".format(
                    config, uid, filter_name))
        except:
            logging.info("\n\nERROR!!!! NOT FIND config: {}, ERRO!!!!!! \n\n Not find uid \n".format(config))
            # logging.info(out_df)
            # raise NotImplementedError
            uid = None
            exp_run = None
    else:
        # if len(out_df) > 1:
        #     uid = out_df.iloc[0]["uid"]
        # else:
        #     uid = out_df["uid"]
        uid = out_df.iloc[0]["uid"]
        exp_run = exp_book.get_run(uid)
    return uid, exp_run




class ExpBook(object):
    """
    This class has all experiment runs. 
    You can filter the `all_df` to get the uid. Then use uid to get runs.
    The definitions of name, state, config, created_at, system_metrics 
            summary and file are aligned with wandb:

    Attributes:
        all_df (pandas.core.frame.DataFrame): all informations of runs
        runs (dict): all runs, {uid (str): run (ExpRun)}
                        uid should be unique.
        filter_name: used to filter the runs.
        sort_value_name: used to filter the runs.
        sort: used to filter the runs.
        ascending: used to filter the runs.
    """

    # def __init__(self, entity, project, root_path, all_df, runs_dict, filter_name, sort_value_name, sort, ascending):
    def __init__(self, entity, project, root_path, filter_name, sort_value_name, sort, ascending):

        self.entity = entity
        self.project = project
        self.runs = {}
        self.root_path = Path(root_path)
        # self.project_path = get_project_path(self.entity, self.project)
        if not self.root_path.exists():
            self.root_path.mkdir(parents=True)
        self.config_groups = {}
        self.config_alias = {}
        self.filter_name = filter_name
        self.sort_value_name = sort_value_name
        self.sort = sort
        self.ascending = ascending

        # for uid in runs_dict.keys():
        #     self.runs[uid] = ExpRun(
        #         runs_dict[uid].entity,
        #         runs_dict[uid].project, runs_dict[uid].id, uid, runs_dict[uid].config,
        #         runs_dict[uid], runs_dict[uid].url, runs_dict[uid].state,
        #         runs_dict[uid].created_at, runs_dict[uid].system_metrics,
        #         runs_dict[uid].summary, runs_dict[uid].file
        #     )

    def set_summary_df(self, summary_df):
        self.all_df = summary_df


    def get_run(self, uid):
        """
        Get run from ExpBook
        """
        if uid not in self.runs:
            self.runs[uid] = download_run(
                wandb_api, self.root_path, entity=self.entity, project=self.project, uid=uid, cache=True)
        return self.runs[uid]



    def add_config_from_alias_run_map(self, alias_run_map):
        for k, v in alias_run_map.items():
            self.add_config(
                config=v["config"], alias=k,
                help_params=v["help_params"], uid=v["uid"],
                find_uid_func=v["find_uid_func"])



    def add_config(self, config, alias=None, uid=None, help_params=None, find_uid_func=None):
        """
        By this method, you can register `find_one_uid`, then we can find
        runs by `find_one_uid()`.
        Here the `config` does not need to have all args in experiment, in case you want to 
        sort the filtered results and choose the highest or lowest one.
        Every config shoule be unique in one ExpPlot.

        Arguments:
            config (dict): all args in config shoule also be a subset of ExpRun.config.
            uid (str):  globel unique identifier for the run, in case you want to compare different 
                        experiment results. `project + '/' + id `.
            help_params (dict): The help_params is used to plot more figures with parameters
                                that are not included in ExpRun.config, in case you want to 
                                make some more powerful expressions.
        """
        if config is None or len(config) == 0:
            config_key = alias
        else:
            config_key = generate_config_key(config)
        if uid is not None:
            exp_run = self.get_run(uid)
        elif find_uid_func is None:
            uid, exp_run = find_one_uid(self, config, help_params, self.filter_name, self.sort_value_name,
                            self.sort, self.ascending)
        elif find_uid_func is not None:
            uid, exp_run = find_uid_func(self, config, help_params, self.filter_name, self.sort_value_name,
                            self.sort, self.ascending)

        self.config_groups[config_key] = {
            "alias": alias,
            "config": config,
            "uid": uid,
            "help_params": help_params,
            "exp_run": exp_run,
            "downloaded": False
        }
        if alias is not None:
            self.config_alias[alias] = config_key
        return config_key, uid, exp_run


    def get_group(self, config_key=None, config=None, alias=None):
        """
        Get group
        """
        if config_key is not None:
            group = self.config_groups[config_key]
        elif config is not None:
            group = self.config_groups[generate_config_key(config)]
        elif alias is not None:
            # logging.info(self.config_groups)
            group = self.config_groups[self.config_alias[alias]]
        else:
            raise NotImplementedError
        return group

    def get_config(self, config_key=None, alias=None):
        assert (config_key is not None) or (alias is not None)
        if config_key is None:
            config_key = self.config_alias[alias] 
        return self.config_groups[config_key]["config"]


    def get_uid(self, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        group = self.get_group(config_key, config, alias)
        return group["uid"]


    def get_help_params(self, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        group = self.get_group(config_key, config, alias)
        return group["help_params"]


    def set_uid(self, uid, config_key=None, config=None, alias=None):
        # assert (config_key is not None) or (config is not None) or (alias is not None)
        group = self.get_group(config_key, config, alias)
        group["uid"] = uid



    def set_help_params(self, help_params, config_key=None, config=None, alias=None):
        group = self.get_group(config_key, config, alias)
        group["help_params"] = help_params

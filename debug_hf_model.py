import random
import pandas as pd
from datasets import load_dataset, load_from_disk, ClassLabel
from itertools import chain
import torch
import torch.distributed as dist
import math
import logging
import numpy as np
import os
import socket
import torchvision

from transformers import (BertConfig, 
                          GPT2Config, 
                          LlamaConfig,
                          LlamaForCausalLM,
                          BertForSequenceClassification, 
                          GPT2LMHeadModel, 
                          Trainer, 
                          TrainingArguments, 
                          DataCollatorForLanguageModeling, 
                          DataCollatorWithPadding,
                          CONFIG_MAPPING,
                          MODEL_MAPPING,
                          AutoConfig,
                          AutoModelForCausalLM,
                          AutoTokenizer,
                          SchedulerType,
                          default_data_collator,
                          get_scheduler,)


dnn="gpt2"
model_dir="/data2/share/zhtang/gpt2"

tokenizer = AutoTokenizer.from_pretrained(dnn, cache_dir=model_dir)
config = GPT2Config.from_pretrained(dnn, cache_dir=model_dir)
print(config)

# config["max_position_embeddings"] = 764
# config["num_hidden_layers"] = 8
# config["hidden_size"] = 512
# config["num_attention_heads"] = 8
# config["num_key_value_heads"] = 8
config.max_position_embeddings = 32
config.num_hidden_layers = 2
config.hidden_size = 32
config.num_attention_heads = 2
config.num_key_value_heads = 2


print(config)
net = AutoModelForCausalLM.from_config(config)
# param = net.transformer.wpe
param = net.lm_head

# net = torchvision.models.alexnet()
# param = net.classifier[-1]


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, "Total-M": total_num/1000000}

# 
number_params = get_parameter_number(net)

from torch.nn import Module

print(f"get_parameter_number: {number_params}")

print(f"type(net): {type(net)}, isinstance(net, torch.Module):{isinstance(net, Module)}")
print(f"net: {net}")


# print(f"type(net.model): {type(net.model)}, isinstance(net.model, torch.Module):{isinstance(net.model, Module)}")
# print(f"type(net.model.layers[0]):{type(net.model.layers[0])}, ")
# print(f"net.model.layers[0]:{net.model.layers[0]}")
# print(f"net.model.layers[0].mlp:{net.model.layers[0].mlp}")

# print(f"====================================")
# print(f"net.model.layers[0].mlp.gate_proj.weight[0,:5]:{net.model.layers[0].mlp.gate_proj.weight[0,:5]}")
# print(f"====================================")

print(f"====================================")
print(f"param.weight[0,:5]:{param.weight[0,:5]}")
print(f"====================================")

from copy import deepcopy
# new_net = net.state_dict()
new_net = deepcopy(net)
def modify_net(net):
    for name, param in net.named_parameters():
        shape = param.data.shape
        param.data = param.data + torch.normal(mean=1.0, std=0.001, size=shape, device=param.data.device)
modify_net(new_net)

print(f"====================================")
print(f"param.weight[0,:5]:{param.weight[0,:5]}")
print(f"====================================")
net.load_state_dict(new_net.state_dict())

print(f"====================================")
print(f"param.weight[0,:5]:{param.weight[0,:5]}")
print(f"====================================")


dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
print(f'os.environ[LOCAL_RANK]: {os.environ["LOCAL_RANK"]}')


nwpernode = 4
selected_gpu = rank % nwpernode
torch.cuda.set_device(selected_gpu)



hostname = socket.gethostname() 
logger = logging.getLogger(hostname)
logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

# logfile = os.path.join('./debug_hf_model', "debug-Rank" + str(rank)+'.log')
# hdlr = logging.FileHandler(logfile)
# formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
# hdlr.setFormatter(formatter)
# logger.addHandler(hdlr) 
logger.info(f"==================") 

# net = 
net.to(selected_gpu)


def is_root():
    return dist.get_rank() == 0 



def allreduce_model_weights_not_inplace(model):
    if isinstance(model, dict):
        avg_params = deepcopy(model)
    else:
        state = model.state_dict()
        avg_params = deepcopy(state)
        # avg_params = deepcopy(model.state_dict())
    # if is_root():
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(avg_net_lm_head.data):{id(avg_params["lm_head.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(lm_head.data):{id(state["lm_head.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wte.data):{id(avg_params["transformer.wte.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wte.data):{id(state["transformer.wte.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wpe.data):{id(avg_params["transformer.wpe.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wpe.data):{id(state["transformer.wpe.weight"].data)}  ')

    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(avg_net_lm_head):{id(avg_params["lm_head.weight"])}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(lm_head):{id(state["lm_head.weight"])}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wte):{id(avg_params["transformer.wte.weight"])}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wte):{id(state["transformer.wte.weight"])}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wpe):{id(avg_params["transformer.wpe.weight"])}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wpe):{id(state["transformer.wpe.weight"])}  ')


    # for name, module in model.name_modules():
    # for name, param in model.name_parameters():
    for name, param in avg_params.items():
        # dist.all_reduce(avg_params[name], op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_params[name], op=dist.ReduceOp.SUM)
        avg_params[name] = avg_params[name] / dist.get_world_size()

    return avg_params


def allreduce_model_weights(model):
    if isinstance(model, dict):
        state_dict = model
    else:
        state_dict = model.state_dict()
    for name, p in state_dict.items():
        dist.all_reduce(state_dict[name].data, op=dist.ReduceOp.AVG)
        # dist.all_reduce(state_dict[name].data, op=dist.ReduceOp.SUM)
    # for name, p in state_dict.items():
    #     state_dict[name].data = state_dict[name].data / dist.get_world_size()

    return state_dict



def param_diversity(model, avg_params=None):
    if avg_params is None:
        if isinstance(model, dict):
            avg_params = deepcopy(model)
        else:
            state = model.state_dict()
            avg_params = deepcopy(state)

        # name = "lm_head.weight"
        # dist.all_reduce(avg_params[name], op=dist.ReduceOp.SUM)
        # avg_params[name] = avg_params[name] / dist.get_world_size()
 
        for name, param in avg_params.items():
            logger.info(f'Before calc {name}  param_diversity , lm_head[10]:{avg_params["lm_head.weight"][0,:5]}  ')
            if "weight" in name:
                # logger.info(f"Comm para: {name}")
                # if is_root():
                #     logger.info(f"BeforeComm para {name} avg: {avg_params[name].norm()}")
                dist.all_reduce(avg_params[name], op=dist.ReduceOp.SUM)
                avg_params[name] = avg_params[name] / dist.get_world_size()
                # dist.all_reduce(avg_params[name], op=dist.ReduceOp.AVG)
                # if is_root():
                #     logger.info(f"AfterComm para {name} avg: {avg_params[name].norm()}")
            logger.info(f'After calc {name}  param_diversity , lm_head[10]:{avg_params["lm_head.weight"][0,:5]}  ')

    # if is_root():
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(avg_net_lm_head.data):{id(avg_params["lm_head.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(lm_head.data):{id(state["lm_head.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wte.data):{id(avg_params["transformer.wte.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wte.data):{id(state["transformer.wte.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wpe.data):{id(avg_params["transformer.wpe.weight"].data)}  ')
    #     logger.info(f'in  allreduce_model_weights_not_inplace , id(transformer.wpe.data):{id(state["transformer.wpe.weight"].data)}  ')



    if is_root():
        # return None, None
        named_diversitys = {}
        # total_diversity = 0.0
        total_diversitys = []
        # raise RuntimeError
        if isinstance(model, dict):
            for name, param in model.items():
                if "weight" in name and ("bn" not in name ):
                    diff = (avg_params[name] - param)
                    # diff = (avg_params[name] - param.data)
                    if param.dtype == torch.long:
                        logger.info(f"!!!!!!!!!!!!!!!!name is type torch.long!!!!!!!!!!!!!!!!")
                        diff = diff.float()
                    named_diversitys[name] = diff.norm() / math.sqrt(diff.numel())
                    named_diversitys[name] = named_diversitys[name].item()
                    total_diversitys.append(named_diversitys[name])
            return named_diversitys, np.mean(total_diversitys)
        else:
            for name, param in model.state_dict().items():
                if "weight" in name and ("bn" not in name ):
                    diff = (avg_params[name] - param)
                    # diff = (avg_params[name] - param.data)
                    # logger.info(f"AfterComm para: {name}, cal diff: {diff}")
                    if param.dtype == torch.long:
                        logger.info(f"!!!!!!!!!!!!!!!!name is type torch.long!!!!!!!!!!!!!!!!")
                        diff = diff.float()
                    named_diversitys[name] = diff.norm() / math.sqrt(diff.numel())
                    named_diversitys[name] = named_diversitys[name].item()
                    total_diversitys.append(named_diversitys[name])
                    # logger.info(f"AfterComm para: {name}, cal diff norm: {named_diversitys[name]}")
            return named_diversitys, np.mean(total_diversitys)
    else:
        return None, None

dist.all_reduce(torch.tensor([1]).to(selected_gpu), op=dist.ReduceOp.SUM)
print(f"====================================")
print(f"rank {rank} param.weight[0,:5]:{param.weight[0,:5]}")
print(f"====================================")

logger.info(f"rank:{rank} Measure diversity")
named_diversitys, total_diversity = param_diversity(net)
if is_root():
    for layer, diversity in named_diversitys.items():
        logger.info(f"rank:{rank} layer: {layer}, diversity: {diversity}")

dist.all_reduce(torch.tensor([1]).to(selected_gpu), op=dist.ReduceOp.SUM)
print(f"====================================")
print(f"rank {rank} param.weight[0,:5]:{param.weight[0,:5]}")
print(f"====================================")

lm_head = net.state_dict()["lm_head.weight"]
# lm_head = net.lm_head.weight
if is_root():
    logger.info(f"id(lm_head):{id(lm_head)}  ")

# avg_net = allreduce_model_weights_not_inplace(net)
avg_net = allreduce_model_weights(net)
# net.load_state_dict(avg_net)

# dist.all_reduce(torch.tensor([1]).to(selected_gpu), op=dist.ReduceOp.SUM)
# logger.info(f"==========    After Avg       ========") 
# print(f"====================================")
# print(f"rank {rank} param.weight[0,:5]:{param.weight[0,:5]}")
# print(f"====================================")
# dist.all_reduce(torch.tensor([1]).to(selected_gpu), op=dist.ReduceOp.SUM)


avg_net_lm_head = avg_net["lm_head.weight"]
if is_root():
    logger.info(f"id(avg_net_lm_head):{id(avg_net_lm_head)}  ")


named_diversitys, total_diversity = param_diversity(net)
if is_root():
    for layer, diversity in named_diversitys.items():
        logger.info(f"rank:{rank} layer: {layer}, diversity: {diversity}")






# huggingface-cli login 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
# HF_ENDPOINT=https://hf-mirror.com python



























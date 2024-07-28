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

param = net.transformer.wpe

# print(f"type(net.model): {type(net.model)}, isinstance(net.model, torch.Module):{isinstance(net.model, Module)}")
# print(f"type(net.model.layers[0]):{type(net.model.layers[0])}, ")
# print(f"net.model.layers[0]:{net.model.layers[0]}")
# print(f"net.model.layers[0].mlp:{net.model.layers[0].mlp}")

# print(f"====================================")
# print(f"net.model.layers[0].mlp.gate_proj.weight[0,:10]:{net.model.layers[0].mlp.gate_proj.weight[0,:10]}")
# print(f"====================================")

print(f"====================================")
print(f"param.weight[0,:10]:{param.weight[0,:10]}")
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
print(f"param.weight[0,:10]:{param.weight[0,:10]}")
print(f"====================================")
net.load_state_dict(new_net.state_dict())

print(f"====================================")
print(f"param.weight[0,:10]:{param.weight[0,:10]}")
print(f"====================================")


dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
print(f'os.environ[LOCAL_RANK]: {os.environ["LOCAL_RANK"]}')


nwpernode = 4
selected_gpu = rank % nwpernode
torch.cuda.set_device(selected_gpu)



hostname = socket.gethostname() 
logger = logging.getLogger(hostname)

logfile = os.path.join('./debug_hf_model_', str(rank)+'.log')
hdlr = logging.FileHandler(logfile)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 

# net = 
net.to(selected_gpu)


def is_root():
    return dist.get_rank() == 0 



def allreduce_model_weights_not_inplace(model):
    if isinstance(model, dict):
        avg_params = deepcopy(model)
    else:
        avg_params = deepcopy(model.state_dict())

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
        # dist.all_reduce(state_dict[name].data, op=dist.ReduceOp.AVG)
        dist.all_reduce(state_dict[name].data, op=dist.ReduceOp.SUM)
    for name, p in state_dict.items():
        state_dict[name].data = state_dict[name].data / dist.get_world_size()

    return state_dict



def param_diversity(model, avg_params=None):
    if avg_params is None:
        if isinstance(model, dict):
            avg_params = deepcopy(model)
        else:
            avg_params = deepcopy(model.state_dict())

        for name, param in avg_params.items():
            if "weight" in name:
                # logger.info(f"Comm para: {name}")
                dist.all_reduce(avg_params[name], op=dist.ReduceOp.SUM)
                avg_params[name] = avg_params[name] / dist.get_world_size()
                # dist.all_reduce(avg_params[name], op=dist.ReduceOp.AVG)
                # logger.info(f"AfterComm para: {name}")


    if is_root():
        # return None, None
        named_diversitys = {}
        # total_diversity = 0.0
        total_diversitys = []
        if isinstance(model, dict):
            for name, param in model.items():
                if "weight" in name and ("bn" not in name ):
                    diff = (avg_params[name] - param.data)
                    # diff = avg_params[name]
                    if param.dtype == torch.long:
                        logger.info(f"!!!!!!!!!!!!!!!!name is type torch.long!!!!!!!!!!!!!!!!")
                        diff = diff.float()
                    # named_diversitys[f"diver/{name}"] = diff.norm() / math.sqrt(diff.numel())
                    # named_diversitys[name] = diff.norm()
                    named_diversitys[name] = diff.norm() / math.sqrt(diff.numel())
                    named_diversitys[name] = named_diversitys[name].item()
                    # named_diversitys[name] = param.data.norm() / math.sqrt(diff.numel())
                    # named_diversitys[f"diver/{name}"] = diff.norm()
                    total_diversitys.append(named_diversitys[name])
            # return named_diversitys, total_diversity
            return named_diversitys, np.mean(total_diversitys)
        else:
            for name, param in model.state_dict().items():
                if "weight" in name and ("bn" not in name ):
                    diff = (avg_params[name] - param.data)
                    # diff = avg_params[name]
                    if param.dtype == torch.long:
                        logger.info(f"!!!!!!!!!!!!!!!!name is type torch.long!!!!!!!!!!!!!!!!")
                        diff = diff.float()
                    # named_diversitys[f"diver/{name}"] = diff.norm() / math.sqrt(diff.numel())
                    # named_diversitys[name] = diff.norm()
                    named_diversitys[name] = diff.norm() / math.sqrt(diff.numel())
                    named_diversitys[name] = named_diversitys[name].item()
                    # named_diversitys[name] = param.data.norm() / math.sqrt(diff.numel())
                    # named_diversitys[f"diver/{name}"] = diff.norm()
                    total_diversitys.append(named_diversitys[name])
            # return named_diversitys, total_diversity
            return named_diversitys, np.mean(total_diversitys)
    else:
        return None, None



logger.info(f"rank:{rank} Measure diversity")
named_diversitys, total_diversity = param_diversity(net)
if is_root():
    for layer, diversity in named_diversitys.items():
        logger.info(f"rank:{rank} layer: {layer}, diversity: {diversity}")



dist.all_reduce(torch.tensor([1]).to(selected_gpu), op=dist.ReduceOp.SUM)








# huggingface-cli login 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
# HF_ENDPOINT=https://hf-mirror.com python



























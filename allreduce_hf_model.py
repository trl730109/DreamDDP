import torch
import torch.distributed as dist
import logging
import os
import socket

from copy import deepcopy

from transformers import (GPT2Config, 
                          AutoModelForCausalLM)

dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
print(f'os.environ[LOCAL_RANK]: {os.environ["LOCAL_RANK"]}')
hostname = socket.gethostname() 
logger = logging.getLogger(hostname)
logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)

selected_gpu = rank % 4
torch.cuda.set_device(selected_gpu)

dnn="gpt2"
model_dir="/data2/share/zhtang/gpt2"

config = GPT2Config.from_pretrained(dnn, cache_dir=model_dir)
print(config)
config.max_position_embeddings = 32
config.num_hidden_layers = 2
config.hidden_size = 32
config.num_attention_heads = 2
config.num_key_value_heads = 2
net = AutoModelForCausalLM.from_config(config)
# param = net.transformer.wpe
param = net.lm_head

net.to(selected_gpu)

# Add 1 to make results more clear.
for name, param in net.named_parameters():
    shape = param.data.shape
    param.data = param.data + torch.normal(mean=1.0, std=0.5, size=shape, device=param.data.device)

def is_root():
    return dist.get_rank() == 0 
def allreduce_model_weights_SUM_DIV(model):
    if isinstance(model, dict):
        avg_params = deepcopy(model)
    else:
        state = model.state_dict()
        avg_params = deepcopy(state)
    for name, param in avg_params.items():
        logger.info(f'Before {name}, lm_head[10]:{avg_params["lm_head.weight"][0,:5]}  ')
        dist.all_reduce(avg_params[name], op=dist.ReduceOp.SUM)
        avg_params[name] = avg_params[name] / dist.get_world_size()
        logger.info(f'After {name}, lm_head[10]:{avg_params["lm_head.weight"][0,:5]}  ')
    return avg_params

def allreduce_model_weights_AVG(model):
    if isinstance(model, dict):
        avg_params = deepcopy(model)
    else:
        state = model.state_dict()
        avg_params = deepcopy(state)
    for name, param in avg_params.items():
        logger.info(f'Before {name}, lm_head[10]:{avg_params["lm_head.weight"][0,:5]}  ')
        dist.all_reduce(avg_params[name], op=dist.ReduceOp.AVG)
        logger.info(f'After {name} , lm_head[10]:{avg_params["lm_head.weight"][0,:5]}  ')

    return avg_params

logger.info("In SUM DIV")
allreduce_model_weights_SUM_DIV(net)
logger.info("In AVG")
allreduce_model_weights_AVG(net)

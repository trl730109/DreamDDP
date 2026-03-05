# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import datetime
import torch
import torch.optim as optim
import numpy as np
import argparse
import json
import os
import re
import settings
from multiprocessing import set_start_method
from collections import defaultdict
import math

from copy import deepcopy

import pytz
import logging
from multiprocessing import set_start_method
from collections import defaultdict

from transformers import BertConfig, GPT2Config, BertForSequenceClassification, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity
from dl_trainer import DLTrainer, _support_datasets, _support_dnns, create_net
from llm_trainer import LLMTrainer, _support_datasets, _support_dnns
from dist_utils import *
import dist_optimizer as dist_optim

# from tensorboardX import SummaryWriter
from compression import compressors
from profiling import benchmark
from mpi4py import MPI

from helpers.exp_path import ExpTool
import layer_group


def _bandwidth_to_int(s):
    """Parse bandwidth string (e.g. '5gbt', '10Gbps') to int for wandb."""
    s = str(s).strip() if s else ''
    m = re.match(r'^(\d+)', s)
    return int(m.group(1)) if m else 0


from layer_group import resnet_groups, resnet_groups_dream, llm_groups_dream, llm_groups_dream_enlarge
comm = MPI.COMM_WORLD
writer = None

GPT2_MAX_GRAD_NORM = 1.0


def count_leaf_layers(model):
    leaf_names = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # This means the module has no children
            leaf_names.append(name)
    return leaf_names

def is_root():
    return dist.get_rank() == 0

def set_seed(seed=3000):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ('true', 'True'):
        return True
    elif isinstance(v, str) and v.lower() in ('false', 'False'):
        return False
    else:
        return v

from settings import logger, formatter


def log_info(msg, *args, **kwargs):
    """仅在 rank 0 上打印，避免多卡重复输出"""
    rank = int(os.environ.get('RANK', 0)) if not dist.is_initialized() else dist.get_rank()
    if rank == 0:
        logger.info(msg, *args, **kwargs)


def clip_grad(model, dnn, max_norm):
    if dnn in ['lstm', 'lstmwt2']:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    elif dnn == 'lstman4':
        torch.nn.utils.clip_grad_norm_(model.parameters(), 400)
    elif dnn in ["gpt2", "bert-base-uncased", "llama2-7B", "llama2-124M", "Qwen2.5-1.5B"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2.0) 



def param_diversity(model, avg_params=None):
    if avg_params is None:
        if isinstance(model, dict):
            avg_params = deepcopy(model)
        else:
            avg_params = deepcopy(model.state_dict())

        # for name, module in model.name_modules():
        # for name, param in model.name_parameters():
        for name, param in avg_params.items():
            if "weight" in name:
                # dist.all_reduce(avg_params[name], op=dist.ReduceOp.SUM)
                # avg_params[name] = avg_params[name] / dist.get_world_size()
                dist.all_reduce(avg_params[name], op=dist.ReduceOp.AVG)

    if is_root():
        named_diversitys = {}
        # total_diversity = 0.0
        total_diversitys = []
        if isinstance(model, dict):
            for name, param in model.items():
                if "weight" in name and ("bn" not in name ):
                    diff = (avg_params[name] - param.data)
                    # diff = avg_params[name]
                    if param.dtype == torch.long:
                        logging.info(f"!!!!!!!!!!!!!!!!name is type torch.long!!!!!!!!!!!!!!!!")
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
                        logging.info(f"!!!!!!!!!!!!!!!!name is type torch.long!!!!!!!!!!!!!!!!")
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


def record_param_diversity_with_period(model, global_iters, nsteps_param_diversity, check_param_diversity):
    # log_info(f'Within the function nsteps_param_diversity: {nsteps_param_diversity}')
    if check_param_diversity and (global_iters % nsteps_param_diversity == 0):
        named_diversitys, total_diversity = param_diversity(model)
        if is_root():
            new_named_diversitys = {}
            for layer, diversity in named_diversitys.items():
                new_named_diversitys[f"diver/{layer}"] = diversity
            ExpTool.record(new_named_diversitys)
            ExpTool.record({"total_diversity": total_diversity})
            log_info(f'Params have diversity: {total_diversity} !!!!!!!!.')
            
def transformer_ssgd(optimizer_name, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None, lr_decay=None,
                         nsteps_param_diversity=None, check_param_diversity=None, args=None, profile=False):
    rank = dist.get_rank()
    log_info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    #print("Assign the gpu ", (rank%nwpernode)+2, " to the rank ", rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = LLMTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer, lr_decay=lr_decay,
                         args=args)
    
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    
    is_sparse = density < 1
    if not is_sparse:
        compressor = None

    if settings.ADAPTIVE_MERGE or settings.ADAPTIVE_SPARSE:
        seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            log_info('layerwise backward times: %s', list(layerwise_times))
            log_info('layerwise backward sizes: %s', list(layerwise_sizes))
        log_info('Bencharmked backward time: %f', np.sum(layerwise_times))
        log_info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None

    log_info('Broadcast parameters....')
    # If using LoRA, only broadcast LoRA parameters
    if args.finetune_type == "lora":
        peft_state_dict = trainer.get_peft_model()
        broadcast_parameters(peft_state_dict, root_rank=0)
        trainer.update_peft_model(peft_state_dict)
        log_info('Broadcast LoRA parameters finished....')
    else:
        broadcast_parameters(trainer.net.state_dict(), root_rank=0)
        log_info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400
        
    optimizer = trainer.optimizer
    
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    log_info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else max(1, iters_per_epoch - 1)
    global_iters = 0
    comm_time_acc = 0.0
    train_time_acc = 0.0
    iter_time_acc = 0.0
    backward_time_acc = 0.0
    fp_bp_time_acc = 0.0

    if profile:
        layer_bp_timestamps = {}
        def add_backward_hook(layer, name):
            def backward_hook(module, grad_input, grad_output):
                torch.cuda.synchronize()
                layer_bp_timestamps[name] = time.time()
            layer.register_full_backward_hook(backward_hook)
        for name, module in trainer.net.named_modules():
            if len(list(module.children())) == 0: 
                add_backward_hook(module, name)
            
    for epoch in range(max_epochs):
        if profile:
            bp_dict = {}
            for name, module in trainer.net.named_modules():
                if len(list(module.children())) == 0: 
                    bp_dict[name] = []
        hidden = None
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
            
        train_epoch_loss = 0.0
        # train_epoch_acc = 0.0
        train_epoch_ppl = 0.0
        result_dict = {}
        
        if profile:
            layer_bp_timestamps = {}
        for i in range(iters_per_epoch//nsteps_update):
            global_iters += 1
            result_dict = {}
            
            iter_start = time.time()
            
            optimizer.zero_grad()
            
            fp_bp_start = time.time()
            for j in range(nsteps_update):
                if j < nsteps_update - 1 and nsteps_update > 1:
                    optimizer.local = True
                else:
                    optimizer.local = False
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
                backward_time_acc += trainer.backwardtime_tmp
            fp_bp_end = time.time()
            fp_bp_time_acc += (fp_bp_end - fp_bp_start)
            
            iter_train_end = time.time()
            train_time_acc += (iter_train_end - iter_start)
            
            iter_comm_start = time.time()
            
            if args.finetune_type == "lora":
                for param in trainer.net.parameters():
                    if param.requires_grad and param.grad is not None:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
            else:
                for param in trainer.net.parameters():
                    if param.requires_grad:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
                        
            iter_comm_time = time.time() - iter_comm_start
            comm_time_acc += iter_comm_time
            
            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            
            train_loss = trainer.loss
            train_ppl = trainer.ppl
            train_epoch_loss += train_loss
            train_epoch_ppl += train_ppl
            
            trainer.update_model()
            iter_end = time.time()
            iter_time = iter_end - iter_start
            times.append(iter_time)
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                log_info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
                
            iter_time_acc += iter_time
            ExpTool.record(result_dict)
            
            total_bp_comm = trainer.backward_acc + comm_time_acc
            
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_ppl": train_ppl, "total train time": train_time_acc,
                        "total comm time": comm_time_acc, "total iteration time": iter_time_acc,
                        "total BP time": trainer.backward_acc, "total FP time": trainer.forward_acc,
                        "total BP and comm time": total_bp_comm, "total fp_bp time": fp_bp_time_acc,
                        "total FP,BP,Comm time": trainer.forward_acc + trainer.backward_acc + comm_time_acc,
                        "bandwidth": _bandwidth_to_int(args.bandwidth)})
            
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()

            if profile:
                previous_time = trainer.backward_stamp
                for name in layer_bp_timestamps:
                    current_stamp = layer_bp_timestamps[name]
                    bp_dict[name].append(current_stamp - previous_time)
                    previous_time = current_stamp
                layer_bp_timestamps = {}
            
        log_info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_ppl, test_loss = trainer.test(epoch)
        result_dict["test_loss"] = test_loss
        result_dict["val_ppl"] = val_ppl
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        # result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_ppl"] = train_epoch_ppl / (iters_per_epoch//nsteps_update)
        
        if profile:
            avg_bp_dict = {}
            for name in bp_dict:
                if ('self_attn.rotary_emb' in name):
                    log_info(f'self_attn.rotary_emb time:{bp_dict[name]}')
                new_name = name
                if name.endswith(".lora_A.default") or name.endswith(".lora_B.default") or name.endswith(".lora_dropout.default"):
                    new_name = name.replace(".default", "")
                avg_bp_dict[new_name] = np.mean(bp_dict[name])
            filename = 'bp' + '_' + dnn + '_' + dataset + '_' + str(nworkers) + 'workers' + '.json'
            save_path = os.path.join('./time', dnn, str(nworkers), args.bandwidth, 'bp', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as file:
                json.dump(avg_bp_dict, file, indent=4, ensure_ascii=False)
            
        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch, "bandwidth": _bandwidth_to_int(args.bandwidth)})
        ExpTool.upload()

def transformer_pipe_sgd(optimizer_name, overlap_scalar, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None, lr_decay=None,
                         nsteps_param_diversity=None, check_param_diversity=None, args=None):
    rank = dist.get_rank()
    log_info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    #torch.manual_seed(rank)
    #print("Assign the gpu ", (rank%nwpernode)+2, " to the rank ", rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = LLMTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer, lr_decay=lr_decay,
                         args=args)
    
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    
    is_sparse = density < 1
    if not is_sparse:
        compressor = None

    if settings.ADAPTIVE_MERGE or settings.ADAPTIVE_SPARSE:
        seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            log_info('layerwise backward times: %s', list(layerwise_times))
            log_info('layerwise backward sizes: %s', list(layerwise_sizes))
        log_info('Bencharmked backward time: %f', np.sum(layerwise_times))
        log_info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None

    log_info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    log_info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400
        
    optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    log_info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else max(1, iters_per_epoch - 1)
    global_iters = 0
    iter_time_acc = 0.0
    backward_time_acc = 0.0
    for epoch in range(max_epochs):
        hidden = None
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
            
        train_epoch_loss = 0.0
        # train_epoch_acc = 0.0
        train_epoch_ppl = 0.0
        result_dict = {}
        
        for i in range(iters_per_epoch//nsteps_update):
            global_iters += 1
            result_dict = {}
            iter_start = time.time()
            optimizer.zero_grad()
            for j in range(nsteps_update):
                if j < nsteps_update - 1 and nsteps_update > 1:
                    optimizer.local = True
                else:
                    optimizer.local = False
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
                backward_time_acc += trainer.backwardtime_tmp

            train_loss = trainer.loss
            train_ppl = trainer.ppl
            # train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            # train_epoch_acc += train_acc
            train_epoch_ppl += train_ppl
            
            trainer.update_model()
            iter_end = time.time()
            iter_time = iter_end - iter_start
            times.append(iter_time)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                log_info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
            iter_time_acc += iter_time
            ExpTool.record(result_dict)
            # pipe_sgd 中通信重叠，无法单独拿到等待时间，这里只计 BP
            total_bp_comm = trainer.backward_acc
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_ppl": train_ppl, "total iteration time": iter_time_acc,
                        "total BP time": trainer.backward_acc, "total FP time": trainer.forward_acc,
                        "total BP and comm time": total_bp_comm,
                        "total FP,BP,Comm time": trainer.forward_acc + trainer.backward_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()    

        #log_info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_ppl, test_loss = trainer.test(epoch)
        result_dict["test_loss"] = test_loss
        result_dict["val_ppl"] = val_ppl
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        # result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_ppl"] = train_epoch_ppl / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()
          
def transformer_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, max_epochs, nwpernode, nsteps_update, tokenizer_name=None, nsteps_localsgd=20, lr_decay = 'step',
             check_param_diversity=None, nsteps_param_diversity=None, args=None, profile=False):
    assert nsteps_localsgd > 1
    set_seed(3000)
    rank = dist.get_rank()
    log_info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    times = []
    trainer = LLMTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=None, num_steps=35, tb_writer=writer,optimizer_name="Adam", lr_decay = lr_decay,
                         args=args)
    log_info(f'rank {rank} Broadcast epoch....')
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    iters_per_epoch = trainer.num_batches_per_epoch
    
    log_info(f'rank {rank} Broadcast parameters....')
    # If using LoRA, only broadcast LoRA parameters
    if args.finetune_type == "lora":
        peft_state_dict = trainer.get_peft_model()
        broadcast_parameters(peft_state_dict, root_rank=0)
        trainer.update_peft_model(peft_state_dict)
        log_info(f'rank {rank} Broadcast LoRA parameters finished....')
    else:
        broadcast_parameters(trainer.net.state_dict(), root_rank=0)
        log_info(f'rank {rank} Broadcast parameters finished....')
    
    global_iters = 0
    train_time_acc = 0.0
    comm_time_acc = 0.0
    iter_time_acc = 0.0
    backward_time_acc = 0.0
    fp_bp_time_acc = 0.0
    display = 1 if iters_per_epoch > 40 else max(1, iters_per_epoch - 1)

    if profile:
        comm_dict = {}
        for name, module in trainer.net.named_modules():
            if len(list(module.children())) == 0: 
                comm_dict[name] = []
    
    for epoch in range(max_epochs):
        trainer.net.train()
        trainer.train_sampler.set_epoch(epoch)
        
        result_dict = {}
        train_epoch_loss = 0.0
        # train_epoch_acc = 0.0  
        train_epoch_ppl = 0.0
        
        # log_info(f' Rank {rank} Enter epochs')
        for j in range(iters_per_epoch):
            global_iters += 1
            iter_start = time.time()
            trainer.zero_grad()
            fp_bp_start = time.time()
            trainer.train(1)
            fp_bp_end = time.time()
            fp_bp_time_acc += (fp_bp_end - fp_bp_start)
            backward_time_acc += trainer.backwardtime_tmp

            train_loss = trainer.loss
            # train_acc = np.mean(trainer.train_acc_top1)
            train_ppl = trainer.ppl
            train_epoch_loss += train_loss
            # train_epoch_acc += train_acc
            train_epoch_ppl += train_ppl
            
            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)

            trainer.update_model()
            iter_train_end = time.time()
            train_time = iter_train_end - iter_start
            train_time_acc += train_time

            if global_iters % nsteps_localsgd == 0:
                comm_start = time.time()
                # If using LoRA, only communicate LoRA parameters
                if args.finetune_type == "lora":
                    # Get LoRA parameters and communicate them
                    peft_state_dict = trainer.get_peft_model()
                    for name, param_tensor in peft_state_dict.items():
                        # name is a parameter key, we map it back to the corresponding module
                        # so that keys are consistent with SGD's bp JSON
                        # e.g. "....lora_A.default.weight" -> "....lora_A.default"
                        if "." in name:
                            module_name = ".".join(name.split(".")[:-1])
                        else:
                            module_name = name

                        if profile:
                            torch.cuda.synchronize()
                            ls = time.time()
                        dist.all_reduce(param_tensor, op=dist.ReduceOp.AVG, async_op=False)
                        if profile:
                            torch.cuda.synchronize()
                            layer_time = time.time() - ls
                            if module_name not in comm_dict:
                                comm_dict[module_name] = []
                            comm_dict[module_name].append(layer_time)

                    # Update the model with synchronized LoRA parameters
                    trainer.update_peft_model(peft_state_dict)
                else:
                    # Full model: communicate all parameters
                    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
                        if len(list(module.children())) == 0:  
                            if profile:
                                torch.cuda.synchronize()
                                ls = time.time()
                            for param in module.parameters():
                                dist.all_reduce(param.data, op=dist.ReduceOp.AVG, async_op=False)
                            if profile:
                                torch.cuda.synchronize()
                                layer_time = time.time() - ls
                                comm_dict[name].append(layer_time)
                        else:
                            pass
                comm_this = time.time() - comm_start
                comm_time_acc += comm_this
            else:
                pass
            
            iter_end = time.time()
            iter_time = iter_end - iter_start
            iter_time_acc += iter_time
            times.append(iter_time)

            if j % display == 0 and j > 0: 
                time_per_iter = np.mean(times)
                log_info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
                
            ExpTool.record(result_dict)
            # ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
            #             "train_acc": train_acc, "train_ppl": train_ppl})
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                          "train_ppl": train_ppl})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()
            
            total_bp_comm = trainer.backward_acc + comm_time_acc
            ExpTool.record({"global_iters": global_iters, "total train time": train_time_acc,
                        "total comm time": comm_time_acc, "total iteration time": iter_time_acc,
                        "total BP time": trainer.backward_acc, "total FP time": trainer.forward_acc,
"total BP and comm time": total_bp_comm, "total fp_bp time": fp_bp_time_acc,
                        "total FP,BP,Comm time": trainer.forward_acc + trainer.backward_acc + comm_time_acc,
                        "bandwidth": _bandwidth_to_int(args.bandwidth)})
            
        val_ppl, test_loss = trainer.test(epoch)
        result_dict["test_loss"] = test_loss
        result_dict["val_ppl"] = val_ppl
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        # result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_ppl"] = train_epoch_ppl / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()
        
    if profile:
        avg_comm_dict = {}
        for name in comm_dict:
            new_name = name
            if name.endswith(".lora_A.default") or name.endswith(".lora_B.default") or name.endswith(".lora_dropout.default"):
                new_name = name.replace(".default", "")

            if len(comm_dict[name]) > 0:
                avg_comm_dict[new_name] = np.mean(comm_dict[name])
        log_info(f'Each layer comm time is {avg_comm_dict}')
        filename = 'comm' + '_' + dnn + '_' + dataset + '_' + str(nworkers) + 'workers' + '.json'
        save_path = os.path.join('./time', dnn, str(nworkers), args.bandwidth, 'comm', filename)
        import json
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as file:
            json.dump(avg_comm_dict, file, indent=4, ensure_ascii=False)

def transformer_pipe_seq_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, max_epochs, nwpernode, nsteps_update, tokenizer_name=None, nsteps_localsgd=20, lr_decay = 'step',
             check_param_diversity=None, nsteps_param_diversity=None, args=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    log_info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = LLMTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=None, num_steps=35, tb_writer=writer,optimizer_name="Adam", lr_decay = lr_decay,
                         args=args)
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))

    if settings.ADAPTIVE_MERGE or settings.ADAPTIVE_SPARSE:
        seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            log_info('layerwise backward times: %s', list(layerwise_times))
            log_info('layerwise backward sizes: %s', list(layerwise_sizes))
        log_info('Bencharmked backward time: %f', np.sum(layerwise_times))
        log_info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    log_info('All the steps before broadcasting params are correct.')
    log_info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    log_info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    log_info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else max(1, iters_per_epoch - 1)

    _handles = {}
    _buffer_params = {}
    _parameter_names = {}

    def is_communicate(__module, gap_iters, begin_comm_iter):
        return __module.sgd_iters % gap_iters == begin_comm_iter

    import copy

    def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
        def hook(*ignore):
            if is_communicate(__param, gap_iters, begin_comm_iter):
                if is_root():
                    log_info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
                                f"layer:{name}/{layer_index}-th, __module: {type(__module)}")

                #buffer_param = copy.deepcopy(__param.data)
                #handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
                # with record_function(f"all_reduce_{name}_{layer_index}"):
                handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
                _handles[__param] = (handle, None, 1)

        return hook

    module_names = count_leaf_layers(trainer.net)
    layer_per_iter = int(len(module_names) / nsteps_localsgd) + 1
    # named_modules = dict(trainer.net.named_modules())
    # layer_per_iter = int(len(named_modules) / nsteps_localsgd) + 1
    log_info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(module_names)} "
                f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if (len(list(module.children()))) == 0: 
            if is_root():
                log_info(f"name: {name}, module id: {id(module)}")
            # log_info(f"name: {name}, module id: {id(module)}")
            for param in module.parameters():
                p_tmp = param.expand_as(param)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(_make_hook(module, param, (len(module_names)-module_names.index(name)-1) // layer_per_iter, gap_iters=nsteps_localsgd, name=name, layer_index=module_names.index(name)))
                grad_accs.append(grad_acc)
        else:
            pass
            
    def update_model_sgd_iters(model, sgd_iters):
        for module in model.modules():
            module.sgd_iters = sgd_iters
            for param in module.parameters():
                param.sgd_iters = sgd_iters

    def synchronize_all_reduced_models():
        for tensor, value in _handles.items():
            handle, ctx, density = value
            handle.wait()

        _handles.clear()
        _buffer_params.clear()

    global_iters = 0
    wait_time_acc = 0.0
    backward_time_acc = 0.0
    train_time_acc = 0.0

    for epoch in range(max_epochs):
        log_info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_ppl = 0.0
        
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
            #_buffer_params = {}
            global_iters += 1
            result_dict = {}
            
            update_model_sgd_iters(trainer.net, i)
            iter_start = time.time()
            optimizer.zero_grad()
            # with record_function("Train_models"):
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
                backward_time_acc += trainer.backwardtime_tmp

            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)

            end_time = time.time()
            synchronize_all_reduced_models()
            wait_time = time.time() - end_time
            wait_time_acc += wait_time
            log_info(f'Global iteration: {global_iters} wait time: {wait_time} total wait time: {wait_time_acc}')

            train_loss = trainer.loss
            # train_acc = np.mean(trainer.train_acc_top1)
            train_ppl = trainer.ppl
            train_epoch_loss += train_loss
            # train_epoch_acc += train_acc
            train_epoch_ppl += train_ppl
            
            trainer.update_model()
            iter_end = time.time()
            train_time = iter_end - iter_start
            times.append(train_time)
            train_time_acc += train_time
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # log_info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            total_bp_comm = trainer.backward_acc + wait_time_acc
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_ppl": train_ppl, "total wait time": wait_time_acc,
                        "total train time": train_time_acc, "total comm time": wait_time_acc,
                        "total BP time": trainer.backward_acc, "total FP time": trainer.forward_acc,
                        "total BP and comm time": total_bp_comm,
                        "total FP,BP,Comm time": trainer.forward_acc + trainer.backward_acc + wait_time_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()  

        val_ppl, test_loss = trainer.test(epoch)
        result_dict["val_ppl"] = val_ppl
        result_dict["test_loss"] = test_loss
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_ppl"] = train_epoch_ppl / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def transformer_full_pipe_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, max_epochs, nwpernode, nsteps_update, tokenizer_name=None, nsteps_localsgd=20, lr_decay = 'step',
             check_param_diversity=None, nsteps_param_diversity=None, args=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    log_info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = LLMTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=None, num_steps=35, tb_writer=writer,optimizer_name="Adam", lr_decay = lr_decay,
                         args=args)
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))

    if settings.ADAPTIVE_MERGE or settings.ADAPTIVE_SPARSE:
        seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            log_info('layerwise backward times: %s', list(layerwise_times))
            log_info('layerwise backward sizes: %s', list(layerwise_sizes))
        log_info('Bencharmked backward time: %f', np.sum(layerwise_times))
        log_info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    log_info('All the steps before broadcasting params are correct.')
    log_info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    log_info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    log_info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else max(1, iters_per_epoch - 1)

    _handles = {}
    _buffer_params = {}
    _parameter_names = {}

    def is_communicate(__module, gap_iters, begin_comm_iter):
        return __module.sgd_iters % gap_iters == begin_comm_iter

    import copy

    def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
        def hook(*ignore):
            if is_communicate(__param, gap_iters, begin_comm_iter):
                if is_root():
                    log_info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
                                f"layer:{name}/{layer_index}-th, __module: {type(__module)}")

                #buffer_param = copy.deepcopy(__param.data)
                #handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
                # with record_function(f"all_reduce_{name}_{layer_index}"):
                handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
                _handles[__param] = (handle, None, 1)

        return hook

    # module_names = count_leaf_layers(trainer.net)
    # layer_per_iter = int(len(module_names) / nsteps_localsgd) + 1
    # # named_modules = dict(trainer.net.named_modules())
    # # layer_per_iter = int(len(named_modules) / nsteps_localsgd) + 1
    # log_info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(module_names)} "
    #             f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if (len(list(module.children()))) == 0: 
            if is_root():
                log_info(f"name: {name}, module id: {id(module)}")
            # log_info(f"name: {name}, module id: {id(module)}")
            group_index = resnet_groups[dnn][nworkers][args.group_num][name]
            for param in module.parameters():
                p_tmp = param.expand_as(param)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(_make_hook(module, param, group_index, gap_iters=args.group_num, name=name, layer_index=layer_index))
                grad_accs.append(grad_acc)
        else:
            pass
            
    def update_model_sgd_iters(model, sgd_iters):
        for module in model.modules():
            module.sgd_iters = sgd_iters
            for param in module.parameters():
                param.sgd_iters = sgd_iters

    def synchronize_all_reduced_models():
        for tensor, value in _handles.items():
            handle, ctx, density = value
            handle.wait()

        _handles.clear()
        _buffer_params.clear()

    global_iters = 0
    wait_time_acc = 0.0
    backward_time_acc = 0.0
    train_time_acc = 0.0

    for epoch in range(max_epochs):
        log_info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_ppl = 0.0
        
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
            #_buffer_params = {}
            global_iters += 1
            result_dict = {}
            
            update_model_sgd_iters(trainer.net, i)
            iter_start = time.time()
            optimizer.zero_grad()
            # with record_function("Train_models"):
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
                backward_time_acc += trainer.backwardtime_tmp

            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)

            end_time = time.time()
            synchronize_all_reduced_models()
            wait_time = time.time() - end_time
            wait_time_acc += wait_time
            log_info(f'Global iteration: {global_iters} wait time: {wait_time} total wait time: {wait_time_acc}')

            train_loss = trainer.loss
            # train_acc = np.mean(trainer.train_acc_top1)
            train_ppl = trainer.ppl
            train_epoch_loss += train_loss
            # train_epoch_acc += train_acc
            train_epoch_ppl += train_ppl
            
            trainer.update_model()
            iter_end = time.time()
            train_time = iter_end - iter_start
            times.append(train_time)
            train_time_acc += train_time
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # log_info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            total_bp_comm = trainer.backward_acc + wait_time_acc
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_ppl": train_ppl, "total wait time": wait_time_acc,
                        "total train time": train_time_acc, "total comm time": wait_time_acc,
                        "total BP time": trainer.backward_acc, "total FP time": trainer.forward_acc,
                        "total BP and comm time": total_bp_comm,
                        "total FP,BP,Comm time": trainer.forward_acc + trainer.backward_acc + wait_time_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()  

        val_ppl, test_loss = trainer.test(epoch)
        result_dict["val_ppl"] = val_ppl
        result_dict["test_loss"] = test_loss
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_ppl"] = train_epoch_ppl / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()
        
        
        
def transformer_dream_ddp(dnn, dataset, data_dir, nworkers, lr, batch_size, max_epochs, nwpernode, nsteps_update, tokenizer_name=None, nsteps_localsgd=20, lr_decay = 'step',
             check_param_diversity=None, nsteps_param_diversity=None, args=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    log_info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = LLMTrainer(rank, nworkers, localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=None, num_steps=35, tb_writer=writer, optimizer_name="Adam", lr_decay=lr_decay,
                         args=args)
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))

    if settings.ADAPTIVE_MERGE or settings.ADAPTIVE_SPARSE:
        seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            log_info('layerwise backward times: %s', list(layerwise_times))
            log_info('layerwise backward sizes: %s', list(layerwise_sizes))
        log_info('Bencharmked backward time: %f', np.sum(layerwise_times))
        log_info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    log_info('All the steps before broadcasting params are correct.')
    log_info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    log_info('Broadcast parameters finished....')

    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    log_info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else max(1, iters_per_epoch - 1)

    # === 优化 1: 使用 List 替代 Dict，极大提升 append 和 iterate 速度 ===
    _handle_list = []
    
    # === 优化 2: 注册列表，用于在 Training Loop 中预计算 Sync Flag ===
    _registered_params = []

    # === 优化 3: 极速 Hook，无计算、无 Log、无查表 ===
    def _make_fast_hook(param):
        def hook(*ignore):
            # 极速检查：只读一个 bool 属性 (在 loop 开始前预计算好)
            if getattr(param, '_do_sync', False):
                handle = dist.all_reduce(param.data, op=dist.ReduceOp.AVG, async_op=True)
                _handle_list.append(handle)
        return hook

    scheduling_path = os.path.join('./time', dnn, str(nworkers), args.bandwidth, 'dreamddp_scheduling.json')
    dreamddp_schedule = None
    dreamddp_schedule_H = None
    if os.path.isfile(scheduling_path):
        with open(scheduling_path) as f:
            raw = json.load(f)
        if 'schedule' in raw:
            dreamddp_schedule_H = raw['H']
            dreamddp_schedule = raw['schedule']
        else:
            dreamddp_schedule = raw
        log_info(f'loaded dreamddp schedule from {scheduling_path}' + (f' (H={dreamddp_schedule_H})' if dreamddp_schedule_H is not None else ''))
    else:
        raise FileNotFoundError(f'dreamddp_scheduling.json not found at {scheduling_path}')

    def schedule_lookup_name(module_name, schedule_dict):
        n = module_name
        if getattr(args, 'finetune_type', 'full') != 'lora':
            return n if n in schedule_dict else None
        for prefix in ('base_model.model.', 'base_model.model.model.', 'base_model.'):
            if not n.startswith(prefix):
                continue
            n2 = n[len(prefix):]
            if n2.endswith('.lora_A.default'):
                n2 = n2[:-len('.lora_A.default')]
            elif n2.endswith('.lora_B.default'):
                n2 = n2[:-len('.lora_B.default')]
            elif n2.endswith('.lora_dropout.default'):
                n2 = n2[:-len('.lora_dropout.default')]
            if n2 in schedule_dict:
                return n2
        return n if n in schedule_dict else None

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if len(list(module.children())) == 0:
            if is_root():
                log_info(f"name: {name}, module id: {id(module)}")
            
            if args.enlarge == False:
                schedule_key = schedule_lookup_name(name, dreamddp_schedule or {})
                if dreamddp_schedule is None or schedule_key is None:
                    continue
                if schedule_key not in dreamddp_schedule:
                    continue

                v = dreamddp_schedule[schedule_key]
                group_index_list = v if isinstance(v, list) else [v]
            else:
                raise ValueError(f'enlarge schedule is not supported yet.')
            
            for name, param in module.named_parameters():
                # 记录调度规则，用于后续预计算
                # 格式: (周期 H, 触发列表)
                # if is_root():
                #     log_info(f"param name: {name}")
                #     log_info("")
                param._dreamddp_schedule_info = (dreamddp_schedule_H, group_index_list)
                _registered_params.append(param)
                
                p_tmp = param.expand_as(param)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                # 注册极速 Hook
                grad_acc.register_hook(_make_fast_hook(param))
                grad_accs.append(grad_acc)
                
    # exit()

    def update_model_sgd_iters(model, sgd_iters):
        for module in model.modules():
            module.sgd_iters = sgd_iters
            for param in module.parameters():
                param.sgd_iters = sgd_iters

    def synchronize_all_reduced_models():
        # 极速同步：遍历 List 直接 Wait
        for handle in _handle_list:
            handle.wait()
        _handle_list.clear() # 清空列表，准备下一轮

    global_iters = 0
    wait_time_acc = 0.0
    train_time_acc = 0.0
    iter_time_acc = 0.0
    sync_flag_time_acc = 0.0
    update_sgd_iters_time_acc = 0.0
    do_sync_loop_time_acc = 0.0
    fp_bp_time_acc = 0.0
    cuda_sync_time_acc = 0.0   # DreamDDP 独有：wait() 前的 torch.cuda.synchronize()，不计入 train

    for epoch in range(max_epochs):
        log_info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_ppl = 0.0
        
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
            
        for i in range(iters_per_epoch // nsteps_update):
            global_iters += 1
            result_dict = {}
            
            iter_start = time.time()
            # schedule 为 0-based（与 dreamddp_scheduling.json 一致），训练中 global_iters 为 1-based，故用 global_iters-1 查表
            sgd_iters_0based = global_iters - 1
            update_model_sgd_iters(trainer.net, sgd_iters_0based)
            t_after_update_sgd_iters = time.time()
            update_sgd_iters_time_acc += (t_after_update_sgd_iters - iter_start)
            
            # === 核心优化: 预计算本轮 Sync Flag ===
            for param in _registered_params:
                gap, begin_list = param._dreamddp_schedule_info
                param._do_sync = (param.sgd_iters % gap) in begin_list
            t_after_do_sync_loop = time.time()
            do_sync_loop_time_acc += (t_after_do_sync_loop - t_after_update_sgd_iters)
            sync_flag_time_acc += (t_after_do_sync_loop - iter_start)
            
            # 口径约定：train = zero_grad + FP+BP + clip_grad（训练步计算）；torch.cuda.synchronize() 单独计，不算 train
            train_start = t_after_do_sync_loop
            optimizer.zero_grad()
            fp_bp_start = time.time()
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
                # backward_time_acc += trainer.backwardtime_tmp
            fp_bp_end = time.time()
            fp_bp_time_acc += (fp_bp_end - fp_bp_start)

            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            train_end = time.time()
            train_time_acc += (train_end - train_start)   # train = zero_grad + fp_bp + clip_grad

            _t0 = time.time()
            torch.cuda.synchronize()
            cuda_sync_time_acc += (time.time() - _t0)
            bp_end_time = time.time()
            
            # 2. 等待通信完成
            synchronize_all_reduced_models()
            
            # 3. 计算纯粹的通信等待时间
            dreamddp_extra_wait_time = time.time() - bp_end_time
            wait_time_acc += dreamddp_extra_wait_time
            
            log_info(f'Global iteration: {global_iters} wait time: {dreamddp_extra_wait_time} total wait time: {wait_time_acc}')
            
            train_loss = trainer.loss
            train_ppl = trainer.ppl
            train_epoch_loss += train_loss
            train_epoch_ppl += train_ppl
            
            trainer.update_model()
            iter_end = time.time()
            iter_time = iter_end - iter_start
            iter_time_acc += iter_time
            times.append(iter_time)
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
                
            # === Total 指标口径 ===
            # total train time = zero_grad + fp_bp + clip_grad（不含 cuda.sync / 通信）
            # total fp_bp time = 仅 FP+BP；dreamddp_cuda_sync_time = wait() 前的 cuda.synchronize()
            total_bp_comm = trainer.backward_acc + wait_time_acc

            ExpTool.record(result_dict)
            ExpTool.record({
                "global_iters": global_iters, 
                "epochs": epoch, 
                "train_loss": train_loss,
                "train_ppl": train_ppl, 
                "total wait time": wait_time_acc,
                "total train time": train_time_acc, 
                "total comm time": wait_time_acc, 
                "total iteration time": iter_time_acc,
                "total BP time": trainer.backward_acc,
                "total FP time": trainer.forward_acc,
                "total BP and comm time": total_bp_comm,
                "total sync flag time": sync_flag_time_acc,
                "total FP,BP,Comm time": trainer.backward_acc + trainer.forward_acc + wait_time_acc,
                "dreamddp_update_sgd_iters_time": update_sgd_iters_time_acc,
                "dreamddp_do_sync_loop_time": do_sync_loop_time_acc,
                "dreamddp_cuda_sync_time": cuda_sync_time_acc,
                "total fp_bp time": fp_bp_time_acc,
                "bandwidth": _bandwidth_to_int(args.bandwidth)
            })
            
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()  

        val_ppl, test_loss = trainer.test(epoch)
        result_dict["val_ppl"] = val_ppl
        result_dict["test_loss"] = test_loss
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch // nsteps_update)
        result_dict["train_epoch_ppl"] = train_epoch_ppl / (iters_per_epoch // nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch, "bandwidth": _bandwidth_to_int(args.bandwidth)})
        ExpTool.upload()
        
def arg_str2bool(args):
    for key in args.__dict__.keys():
        args.__dict__[key] = str2bool(args.__dict__[key])





if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--nwpernode', type=int, default=1, help='Number of workers per node')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=_support_datasets, help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--saved-dir', type=str, default='.', help='Specify the saved weights or gradients root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--lr_decay', type=str, default=None,help='learning rate decay methods, choosing from None,cosine,step')
    parser.add_argument('--max-epochs', type=int, default=settings.MAX_EPOCHS, help='Default maximum epochs to train')
    parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    parser.add_argument('--num-steps', type=int, default=35)
    parser.add_argument('--compressor', type=str, default='topk', choices=compressors.keys(), help='Specify the compressors if density < 1.0')
    parser.add_argument('--density', type=float, default=1, help='Density for sparsification')
    parser.add_argument('--threshold', type=int, default=0, help='Specify the threshold for gradient merging')
    parser.add_argument('--momentum-correction', type=int, default=0, help='Set it to 1 to turn on momentum_correction for TopK sparsification, default is 0')
    parser.add_argument('--strategy', type=str, default='average', help='gradient averaging strategies, choosing from ties, ties_max, average, overlap')
    parser.add_argument('--overlap_scalar', type=float, default=2, help='Overlap scalar for TopK sparsification, default is 0.1')
    parser.add_argument('--nsteps_localsgd', type=int, default=10)
    parser.add_argument('--optimizer_name',type=str, default=None, help='Optimizer used in the training, default to be SGD.')
    parser.add_argument('--adam_beta1',type=float, default=0.9, help='.')
    parser.add_argument('--adam_beta2',type=float, default=0.999, help='.')
    parser.add_argument('--weight_decay',type=float, default=0.0001, help='.')

    parser.add_argument('--enlarge', type = str, default='False',help='')
    parser.add_argument('--model_dir', type=str, default='./model', help='')
    parser.add_argument('--load_pretrain', type=str, default='False', help='')
    # 量化加载相关参数（与 fault_dist_trainer.py 保持一致）
    parser.add_argument('--load_quantization', type=str, default='no', help='')

    parser.add_argument('--interface', default='eno0', help='Network interface, choosing from eno0-1G, ens5f0-10G')
    parser.add_argument('--alg', type=str,default='localsgd',help='Algorithms including desync, sgd, localsgd, layerwise.')
    parser.add_argument('--local-rank', type=int, default=0,help='local rank for distributed training')
    parser.add_argument('--group_num',type=int, default='6', help='Number of iterations to achieve full synchronziation in full_pipe_Seq.')
    parser.add_argument('--config_name', type=str, default='', help='Model configurations.')
    parser.add_argument('--model_name_or_path', type=str,default='',help='Local model path for GPT or Bert.')
    parser.add_argument('--training_type', type=str,default='pretrain',help='training type, pretrain or lora finetunie.')
    parser.add_argument('--finetune_type', type=str,default='full',help='training type, pretrain or lora finetunie.')
    parser.add_argument('--peft_lora_r', type=int, default=8, help='LoRA rank parameter.')
    parser.add_argument('--peft_lora_alpha', type=int, default=16, help='LoRA alpha parameter.')
    # Check model divergence
    parser.add_argument('--check_param_diversity', type=str, default="False")
    parser.add_argument('--nsteps_param_diversity', type=int, default=5)
    parser.add_argument('--profile', type=str, default="False", help='Enable profiling: bp (layer backward time), save to json')

    # wandb, exp record related
    parser.add_argument("--wandb_offline", type=str, default="True")
    parser.add_argument("--wandb_console", type=str, default="False")
    parser.add_argument("--wandb_entity", type=str, default="your-wandb-entity")
    parser.add_argument("--wandb_key", type=str, default=None)

    parser.add_argument("--exp_abs_path", type=str, default=".")
    parser.add_argument("--project_name", type=str, default="your-wandb-project")
    parser.add_argument("--exp_name", type=str, default="OneShot-FL")
    parser.add_argument("--override_cmd_args", action="store_true")
    parser.add_argument("--tag", type=str, default="debug")
    parser.add_argument("--exp_tool_init_sub_dir", type=str, default="no")
    
    parser.add_argument("--bandwidth", type=str, default="10Gbps", help='Bandwidth for the network')
    
    parser.add_argument("--enable_wandb", type=str, default="False")
    args = parser.parse_args()
    arg_str2bool(args)
    batch_size = args.batch_size * args.nsteps_update
    momentum_correction = args.momentum_correction != 0
    prefix = args.alg + '-' + 'lr_decay_' + args.lr_decay

  
    beijing_tz = pytz.timezone('Asia/Shanghai')

    logdir = '%s' % (datetime.datetime.now(beijing_tz).strftime("%m-%d-%H:%M")) + '-' + prefix

    directory_path = os.path.join('./test', args.alg, args.dnn)

    relative_path = os.path.join(directory_path, logdir)

    print(relative_path)

    gradient_relative_path = None 
    utils.create_path(relative_path)
    if settings.LOGGING_GRADIENTS:
        gradient_relative_path = '%s/gradients/%s'%(args.saved_dir, logdir)
        utils.create_path(gradient_relative_path)
    rank = 0
    #set_start_method('spawn')
    if args.nworkers > 1:
        # dist.init_process_group(backend='nccl')
        # rank = dist.get_rank()
        # os.environ['NCCL_DEBUG'] = 'INFO'
        # os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        # os.environ['NCCL_DEBUG'] = 'TRACE'
        
        # os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
        # if args.interface == 'eno0':
        #     os.environ['NCCL_SOCKET_IFNAME'] = 'eno0' #,ens5f0
        # elif args.interface == 'ens5f0':
        #     os.environ['NCCL_SOCKET_IFNAME'] = 'ens5f0'
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0' #,ens5f0
        log_info(f"Before init_process_group, rank env: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
        os.environ['WANDB_MODE'] = 'offline'
        #log_info(f"NCCL_SOCKET_IFNAME is set to: {os.environ.get('NCCL_SOCKET_IFNAME')}")
        dist.init_process_group(backend='nccl', init_method='env://')
        # args.local-rank = int(os.environ['LOCAL_RANK'])
        log_info(f"After init_process_group, rank env: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        rank = dist.get_rank()
        #log_info(f'The rank is consistent {rank == args.local_rank}')
        #print("The Torch.distributed is initialized by rank: ", rank)
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = None #SummaryWriter(tb_runs)

    set_seed(3000)
    ExpTool.init(args, dist)    
    os.environ['WANDB_MODE'] = 'offline'
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    log_info('Configurations: %s', args)


    if (args.alg == 'transformer_localsgd'):
        log_info("Alg used: transformer training.")
        transformer_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.max_epochs, args.nwpernode, args.nsteps_update, tokenizer_name=None, nsteps_localsgd=args.nsteps_localsgd, lr_decay=args.lr_decay, 
             check_param_diversity=args.check_param_diversity, nsteps_param_diversity=args.nsteps_param_diversity, args=args, profile=args.profile)
        
    elif (args.alg == 'transformer_sgd'):
        log_info("Alg used: transformer_sgd.")
        transformer_ssgd(args.optimizer_name, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, lr_decay=args.lr_decay,
                        check_param_diversity=args.check_param_diversity, nsteps_param_diversity=args.nsteps_param_diversity, args=args, profile=args.profile)
        
    elif (args.alg == 'transformer_pipe_sgd'):
        log_info("Alg used: transformer_pipe_sgd.")
        transformer_pipe_sgd(args.optimizer_name, args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, lr_decay=args.lr_decay,
                        check_param_diversity=args.check_param_diversity, nsteps_param_diversity=args.nsteps_param_diversity, args=args)
            
    elif (args.alg == 'transformer_pipe_seq_localsgd'):
        log_info("Alg used: transformer_pipe_seq_localsgd.")
        transformer_pipe_seq_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size,args.max_epochs, args.nwpernode,args.nsteps_update, tokenizer_name=None, nsteps_localsgd=args.nsteps_localsgd, lr_decay=args.lr_decay, 
             check_param_diversity=args.check_param_diversity, nsteps_param_diversity=args.nsteps_param_diversity, args=args)
        
    elif (args.alg == 'transformer_full_pipe_localsgd'):
        log_info("Alg used: transformer_full_pipe_localsgd.")
        transformer_full_pipe_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size,args.max_epochs, args.nwpernode,args.nsteps_update, tokenizer_name=None, nsteps_localsgd=args.nsteps_localsgd, lr_decay=args.lr_decay, 
             check_param_diversity=args.check_param_diversity, nsteps_param_diversity=args.nsteps_param_diversity, args=args)
        
    elif (args.alg == 'transformer_dream_ddp'):
        log_info("Alg used: transformer_dream_ddp.")
        transformer_dream_ddp(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size,args.max_epochs, args.nwpernode,args.nsteps_update, tokenizer_name=None, nsteps_localsgd=args.nsteps_localsgd, lr_decay=args.lr_decay, 
             check_param_diversity=args.check_param_diversity, nsteps_param_diversity=args.nsteps_param_diversity, args=args)

    ExpTool.finish(args)

    #local_sgd_with_dist(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)



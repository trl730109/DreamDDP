# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import datetime
import torch
import torch.optim as optim
import numpy as np
import argparse
import os
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
from layer_group import resnet_groups, resnet_groups_dream
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

def clip_grad(model, dnn, max_norm):
    if dnn in ['lstm', 'lstmwt2']:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    elif dnn == 'lstman4':
        torch.nn.utils.clip_grad_norm_(model.parameters(), 400)
    elif dnn in ["gpt2", "bert-base-uncased", "llama2-7B", "llama2-124M"]:
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
    if check_param_diversity and (global_iters % nsteps_param_diversity == 0):
        named_diversitys, total_diversity = param_diversity(model)
        if is_root():
            new_named_diversitys = {}
            for layer, diversity in named_diversitys.items():
                new_named_diversitys[f"diver/{layer}"] = diversity
            ExpTool.record(new_named_diversitys)
            ExpTool.record({"total_diversity": total_diversity})
            logger.info(f'Params have diversity: {total_diversity} !!!!!!!!.')



def ssgd(optimizer_name, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None, lr_decay=None,
             check_param_diversity=None, nsteps_param_diversity=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    #print("Assign the gpu ", (rank%nwpernode)+2, " to the rank ", rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer, lr_decay=lr_decay, args=args)
    
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
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None

    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400
        
    optimizer = trainer.optimizer
    
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1
    global_iters = 0
    comm_time_acc = 0.0
    train_time_acc = 0.0
    iter_time_acc = 0.0
    backward_time_acc = 0.0
    

    # layer_bp_timestamps = {}
    # def add_backward_hook(layer, name):
    #     def backward_hook(module, grad_input, grad_output):
    #         # Record the current time as the end time for this layer's backward computation
    #         torch.cuda.synchronize()
    #         layer_bp_timestamps[name] = time.time()
    #     layer.register_full_backward_hook(backward_hook)
    # for name, module in trainer.net.named_modules():
    #     if len(list(module.children())) == 0: 
    #         add_backward_hook(module, name)
            
    for epoch in range(max_epochs):
        bp_dict = {}
        for name, module in trainer.net.named_modules():
            if len(list(module.children())) == 0: 
                bp_dict[name] = []
        hidden = None
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
            
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        result_dict = {}
        layer_bp_timestamps = {}
        for i in range(iters_per_epoch//nsteps_update):
            global_iters += 1
            result_dict = {}
            s = time.time()
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
            
            #logger.info(trainer.net.named_modules())
            backward_time_acc += trainer.backwardtime_tmp
            #logger.info(f'Global iteration: {global_iters} backward time: {trainer.backwardtime_tmp} train time: {train_time} \n wait time: {wait_time} total wait time: {wait_time_acc}')
            train_time_acc += (time.time() - s)
            torch.cuda.synchronize()
            comm_s = time.time()
            for param in trainer.net.parameters():
                if param.requires_grad:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
                    #param.grad.data /= dist.get_world_size()
            torch.cuda.synchronize()
            comm_time_acc += (time.time() - comm_s)
            
            # optimizer.synchronize()
            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            # if dnn in ['lstm', 'lstmwt2']:
            #     optimizer.synchronize()
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     optimizer.synchronize()
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_epoch_acc
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
            iter_time_acc += time.time() - s
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "total train time": train_time_acc,
                        "total comm time": comm_time_acc, "total iteration time": iter_time_acc})
            #logger.info(f'check_param_diversity {check_param_diversity}')
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()

            # previous_time = trainer.backward_stamp
            # for name in layer_bp_timestamps:
            #     current_stamp = layer_bp_timestamps[name]
            #     bp_dict[name].append(current_stamp - previous_time)
            #     previous_time = current_stamp
            # layer_bp_timestamps = {}
            
        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)
        # avg_bp_dict = {}
        # for name in bp_dict:
        #     avg_bp_dict[name] = np.mean(bp_dict[name])
        # logger.info(f'Avg bp time for each layer: {avg_bp_dict}')
        
        # filename = 'bp' + '_' + dnn + '_' + dataset + '_' + str(nworkers) + 'workers' + '.json'
        # save_path = os.path.join('./time/new_bp/', filename)
        # import json
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # with open(save_path, 'w') as file:
        #     json.dump(avg_bp_dict, file, indent=4)
            
        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def ssgd_with_pipe(optimizer_name, overlap_scalar, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None, lr_decay=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    #torch.manual_seed(rank)
    #print("Assign the gpu ", (rank%nwpernode)+2, " to the rank ", rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer, lr_decay=lr_decay, args=args)
    
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
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None

    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400
        
    optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1
    global_iters = 0
    iter_time_acc = 0.0
    for epoch in range(max_epochs):
        hidden = None
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
            
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        result_dict = {}
        
        for i in range(iters_per_epoch//nsteps_update):
            global_iters += 1
            result_dict = {}
            s = time.time()
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
 
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
            iter_time_acc += time.time() - s
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "total iteration time": iter_time_acc})
            ExpTool.upload()    
            
        #logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def localsgd_measure(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, lr_decay=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name,lr_decay=lr_decay, args=args)
    
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    # trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
    # trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    #logger.info(f'Successfully set the epoch and iteration to {init_epoch} and {init_iter}')
    
    is_sparse = density < 1
    if not is_sparse:
        compressor = None

    if settings.ADAPTIVE_MERGE or settings.ADAPTIVE_SPARSE:
        seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    logger.info('All the steps before broadcasting params are correct.')
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1

    global_iters = 0
    layer_timestamps = {}
    backward_dict = {}
    def add_backward_hook(layer, name):
        def backward_hook(module, grad_input, grad_output):
            # Record the current time as the end time for this layer's backward computation
            layer_timestamps[name] = time.time()
        layer.register_full_backward_hook(backward_hook)
        
    def calculate_backward_times(start_backward_time):
    # Obtain layer names in the order of backward computation
        layer_names = list(reversed([name for name, _ in trainer.net.named_modules()]))
        backward_times = {}
        previous_time = start_backward_time  # This should be set to the time when backward starts
        for name in layer_names:
            current_time = layer_timestamps.get(name, previous_time)
            backward_times[name] = current_time - previous_time
            previous_time = current_time
        return backward_times

    for name, module in trainer.net.named_modules():
        add_backward_hook(module, name)

    total_list = {name:[] for name,_ in trainer.net.named_modules()}
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0    

        for i in range(iters_per_epoch//nsteps_update):
            global_iters += 1
            result_dict = {}
            s = time.time()
            optimizer.zero_grad()
            
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)

            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            # if dnn in ['lstm', 'lstmwt2']:
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
                
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            bk_list = calculate_backward_times(trainer.backward_stamp)
            for name in bk_list.keys():
                total_list[name].append(bk_list[name])
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)

            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
 
    backward_dict = {}
    back_sum = 0.0
    for name in total_list.keys():
        backward_dict[name] = np.mean(total_list[name])
        back_sum += np.mean(total_list[name])
    logger.info(f'Sum of layer backward is {back_sum}')
    logger.info(f'Each layer backward time {backward_dict}')

def localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, lr_decay=None,
             check_param_diversity=None, nsteps_param_diversity=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name,lr_decay=lr_decay, args=args)
    
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    # trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
    # trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    #logger.info(f'Successfully set the epoch and iteration to {init_epoch} and {init_iter}')
    
    is_sparse = density < 1
    if not is_sparse:
        compressor = None

    if settings.ADAPTIVE_MERGE or settings.ADAPTIVE_SPARSE:
        seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
        layerwise_times = comm.bcast(layerwise_times, root=0)
        if rank == 0:
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    logger.info('All the steps before broadcasting params are correct.')
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1

    global_iters = 0
    train_time_acc = 0.0
    backward_time_acc = 0.0
    comm_time_acc = 0
    iteration_time_acc = 0
    log_dir = f'./logs/profiler_rank_{rank}'
    os.makedirs(log_dir, exist_ok=True)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #              schedule=torch.profiler.schedule(wait=2, warmup=1, active=7,repeat=3),
    #              #on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
    #              record_shapes=True,
    #              with_stack=False) as prof:
    
    # comm_dict = {}
    # for name, module in trainer.net.named_modules():
    #     if len(list(module.children())) == 0: 
    #         comm_dict[name] = []
            
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0    
        backward_list = []
                
        for i in range(iters_per_epoch//nsteps_update):
            global_iters += 1
            result_dict = {}
            s = time.time()
            optimizer.zero_grad()
            
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                        _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            # if dnn in ['lstm', 'lstmwt2']:
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
                
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            backward_time_acc += trainer.backwardtime_tmp
            backward_list.append(trainer.backwardtime_tmp)
            # if(trainer.backwardtime_tmp > 0.5):
            #     logger.info(f'The backward time is abnormal.')
            #     logger.info(f'iteration No.{global_iters} out of total {iters_per_epoch} iterations. Backward Time: {trainer.backwardtime_tmp}')
            trainer.backwardtime_tmp = 0.0
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
                
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "Backward_time": trainer.backwardtime_tmp})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()
                       
            # with record_function("communication"):
            if global_iters % nsteps_localsgd == nsteps_localsgd - 1:
                #logger.info(f'at Iteration {global_iters} do communication.')
                start = time.time()
                for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
                    if len(list(module.children())) == 0:  
                        # torch.cuda.synchronize()
                        # ls = time.time()
                        for param in module.parameters():
                            dist.all_reduce(param.data, op=dist.ReduceOp.AVG, async_op=False)
                        # torch.cuda.synchronize()
                        # layer_time = time.time() - ls
                        # comm_dict[name].append(layer_time)
                        # for state in trainer.optimizer.state.values():
                        #     for k, v in state.items():
                        #         if isinstance(v, torch.Tensor):
                        #             dist.all_reduce(v, op=dist.ReduceOp.AVG)
                    else:
                        pass
                comm_time = time.time() - start
                comm_time_acc += comm_time
                # optimizer_state = trainer.optimizer.state_dict()
                
            else:
                pass
            iteration_time_acc += (time.time() - s)
            ExpTool.record({"global_iters": global_iters, "iteration time": iteration_time_acc, "total train time": train_time_acc,
                        "total comm time": comm_time_acc, "avg backward time": (backward_time_acc / global_iters), "total backward time": backward_time_acc})
            # prof.step()  # Update profiler for each iteration

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)
        
        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()
    # trace_path = os.path.join(log_dir, f'trace_epoch_1.json')
    # prof.export_chrome_trace(trace_path)
    # print(f"Trace saved to {trace_path}")s
    ##########################################
    # avg_comm_dict = {}
    # for name in comm_dict:
    #     avg_comm_dict[name] = np.mean(comm_dict[name])
    # logger.info(f'Each layer comm time is {avg_comm_dict}')
    
    # filename = 'comm' + '_' + dnn + '_' + dataset + '_' + str(nworkers) + 'workers' + '.json'
    # save_path = os.path.join('./time/comm/', filename)
    # import json
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # with open(save_path, 'w') as file:
    #     json.dump(avg_comm_dict, file, indent=4)

def pipe_seq_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, lr_decay=None,
             check_param_diversity=None, nsteps_param_diversity=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name, lr_decay=lr_decay, args=args)

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
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    logger.info('All the steps before broadcasting params are correct.')
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1

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
                    logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
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
    logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(module_names)} "
                f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if (len(list(module.children()))) == 0: 
            if is_root():
                logger.info(f"name: {name}, module id: {id(module)}")
            # logger.info(f"name: {name}, module id: {id(module)}")
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
    # log_dir = f'./logs/pipe_seq_localsgd/profiler_rank_{rank}'
    # os.makedirs(log_dir, exist_ok=True)
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #              schedule=torch.profiler.schedule(wait=1, warmup=1, active=8),
    #             #  on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    #              record_shapes=True,
    #              with_stack=False) as prof:
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
            #_buffer_params = {}
            global_iters += 1
            result_dict = {}
            
            update_model_sgd_iters(trainer.net, i)
            s = time.time()
            optimizer.zero_grad()
            # with record_function("Train_models"):
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)

            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            # if dnn in ['lstm', 'lstmwt2']:
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            end_time = time.time()
            
            synchronize_all_reduced_models()
            wait_time = time.time() - end_time
            wait_time_acc += wait_time
            logger.info(f'Global iteration: {global_iters} wait time: {wait_time} total wait time: {wait_time_acc}')

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            backward_time_acc += trainer.backwardtime_tmp
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "total wait time": wait_time_acc, "total backward time":backward_time_acc, 
                        "total train time": train_time_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()
    # trace_path = os.path.join(log_dir, f'pipe_seq_localsgd.json')
    # prof.export_chrome_trace(trace_path)
    # print(f"Trace saved to {trace_path}") 
    # logger.info(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    
def pipe_seq_localsgd_warmup(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, lr_decay=None,
             check_param_diversity=None, nsteps_param_diversity=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name, lr_decay=lr_decay, args=args)

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
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    logger.info('All the steps before broadcasting params are correct.')
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1

    
    # warmup_optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    # trainer.update_optimizer(warmup_optimizer)
    
    global_iters = 0
    wait_time_acc = 0.0
    backward_time_acc = 0.0
    train_time_acc = 0.0
    warmup_epoches = int(max_epochs * 0.2)
    
    # for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
    #     if len(list(module.children())) == 0:  
    #         for param in module.parameters():
    #             dist.all_reduce(param.data, op=dist.ReduceOp.AVG, async_op=False)

    # #Finish the warmup training
    # model_dict = trainer.net.state_dict()
    # net,_ = create_net(trainer.num_classes, dnn=trainer.dnn, dataset=trainer.dataset)
    # trainer.net = net.cuda()

    # trainer.net.load_state_dict(model_dict)
    # logger.info('Broadcast parameters....')
    # broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    # logger.info('Broadcast parameters finished....')
    
    # for key, tensor in model_dict.items():
    #     model_dict[key] = tensor.to('cpu')
    
    
    _handles = {}
    _buffer_params = {}

    def is_communicate(__module, gap_iters, begin_comm_iter):
        return __module.sgd_iters % gap_iters == begin_comm_iter

    import copy

    def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
        def hook(*ignore):
            if is_communicate(__param, gap_iters, begin_comm_iter):
                if is_root():
                    logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
                                f"layer:{name}/{layer_index}-th, __module: {type(__module)}")

                # with record_function(f"all_reduce_{name}_{layer_index}"):
                handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
                _handles[__param] = (handle, None, 1)

        return hook

    module_names = count_leaf_layers(trainer.net)
    layer_per_iter = int(len(module_names) / nsteps_localsgd) + 1

    logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(module_names)} "
                f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if (len(list(module.children()))) == 0: 
            if is_root():
                logger.info(f"name: {name}, module id: {id(module)}")
            # logger.info(f"name: {name}, module id: {id(module)}")
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
    # max_epochs- warmup_epoches
    for epoch in range(5):
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
    
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
            #_buffer_params = {}
            global_iters += 1
            result_dict = {}
            
            update_model_sgd_iters(trainer.net, i)
            s = time.time()
            optimizer.zero_grad()
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)

            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            # if dnn in ['lstm', 'lstmwt2']:
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            for name, param in trainer.net.named_parameters():
                if param.grad is None:
                    logger.info(f"Gradient for {name}: None")
                    
            # for name, param in trainer.net.named_parameters():
            #     if param.grad is not None:
            #         logger.info(f"Gradient for {name}: {param.grad}")
            #     else:
            #         logger.warning(f"Gradient for {name} is None.")
                   
            end_time = time.time()
            synchronize_all_reduced_models()
            wait_time = time.time() - end_time
            wait_time_acc += wait_time
            logger.info(f'Global iteration: {global_iters} Sync hooks triggered')

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            backward_time_acc += trainer.backwardtime_tmp
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "total wait time": wait_time_acc, "total backward time":backward_time_acc, 
                        "total train time": train_time_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()
        
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if len(list(module.children())) == 0:  
            for param in module.parameters():
                dist.all_reduce(param.data, op=dist.ReduceOp.AVG, async_op=False)

    #Finish the warmup training
    model_dict = trainer.net.state_dict()
    net,_ = create_net(trainer.num_classes, dnn=trainer.dnn, dataset=trainer.dataset)
    trainer.net = net.cuda()

    trainer.net.load_state_dict(model_dict)
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')
    optimzier_state_dict = trainer.optimizer.state_dict()
    warmup_optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    trainer.update_optimizer(warmup_optimizer)
    trainer.optimizer.load_state_dict(optimzier_state_dict)
    
    # for epoch in range(max_epochs- warmup_epoches, max_epochs):
    for epoch in range(5, 40):
        hidden = None
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
    
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
            global_iters += 1
            result_dict = {}
            s = time.time()
            optimizer.zero_grad()
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)

            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            # if dnn in ['lstm', 'lstmwt2']:
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            backward_time_acc += trainer.backwardtime_tmp
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "total wait time": wait_time_acc, "total backward time":backward_time_acc, 
                        "total train time": train_time_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def test(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, lr_decay=None, group_num=6,
             check_param_diversity=None, nsteps_param_diversity=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name, lr_decay=lr_decay, args=args)

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
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    logger.info('All the steps before broadcasting params are correct.')
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1

    _handles = {}
    _buffer_params = {}


    def is_communicate(__module, gap_iters, begin_comm_iter):
        return __module.sgd_iters % gap_iters == begin_comm_iter

    import copy

    def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
        def hook(*ignore):
            if is_communicate(__param, gap_iters, begin_comm_iter):
                if is_root():
                    logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
                                f"layer:{name}/{layer_index}-th, __module: {type(__module)}")

                #buffer_param = copy.deepcopy(__param.data)
                #handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
                handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
                _handles[__param] = (handle, None, 1)

        return hook

    # module_names = count_leaf_layers(trainer.net)

    # layer_per_iter = int(len(module_names) / nsteps_localsgd) + 1
    # fewer_iters = nsteps_localsgd - len(module_names) % nsteps_localsgd
    # division_index = layer_per_iter * fewer_iters

    # logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(named_modules)} "
    #             f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if (len(list(module.children()))) == 0: 
            if is_root():
                logger.info(f"name: {name}, module id: {id(module)}")

            group_index = resnet_groups[dnn][nworkers][group_num][name]
            # group_index = resnet_groups[group_num][name]
            for param in module.parameters():
                p_tmp = param.expand_as(param)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(_make_hook(module, param, group_index, gap_iters=group_num, name=name, layer_index=layer_index))

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
        #logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
            
        for i in range(iters_per_epoch//nsteps_update):
            #_buffer_params = {}
            global_iters += 1
            result_dict = {}
            
            update_model_sgd_iters(trainer.net, i)
            s = time.time()
            optimizer.zero_grad()
            
            # with record_function("Train_models"):
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            
            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            # if dnn in ['lstm', 'lstmwt2']:
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
                
            end_time = time.time()

            synchronize_all_reduced_models()
            wait_time = time.time() - end_time
            wait_time_acc += wait_time
                # logger.info(f'Global iteration: {global_iters} wait time: {wait_time} total wait time: {wait_time_acc}')
                
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            backward_time_acc += trainer.backwardtime_tmp
            
            logger.info(f'Global iteration: {global_iters} backward time: {trainer.backwardtime_tmp} train time: {train_time} \n wait time: {wait_time} total wait time: {wait_time_acc}')
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "total wait time": wait_time_acc , "total backward time":backward_time_acc, 
                        "total train time": train_time_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def dream_ddp(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, lr_decay=None, group_num=6,
             check_param_diversity=None, nsteps_param_diversity=None):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name, lr_decay=lr_decay, args=args)

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
            logger.info('layerwise backward times: %s', list(layerwise_times))
            logger.info('layerwise backward sizes: %s', list(layerwise_sizes))
        logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
        logger.info('Model size: %d', np.sum(layerwise_sizes))
    else:
        seq_layernames, layerwise_times, layerwise_sizes = None, None, None
    logger.info('All the steps before broadcasting params are correct.')
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')

    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1

    _handles = {}
    _buffer_params = {}


    def is_communicate(__module, gap_iters, begin_comm_iter_list):
        return ((__module.sgd_iters % gap_iters) in begin_comm_iter_list)

    import copy

    def _make_hook(__module, __param, begin_comm_iter_list, gap_iters, name, layer_index):
        def hook(*ignore):
            if is_communicate(__param, gap_iters, begin_comm_iter_list):
                if is_root():
                    logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter_list:{begin_comm_iter_list}, communicated successfully "
                                f"layer:{name}/{layer_index}-th, __module: {type(__module)}")

                #buffer_param = copy.deepcopy(__param.data)
                #handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
                handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
                _handles[__param] = (handle, None, 1)

        return hook

    # module_names = count_leaf_layers(trainer.net)

    # layer_per_iter = int(len(module_names) / nsteps_localsgd) + 1
    # fewer_iters = nsteps_localsgd - len(module_names) % nsteps_localsgd
    # division_index = layer_per_iter * fewer_iters

    # logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(named_modules)} "
    #             f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if (len(list(module.children()))) == 0: 
            if is_root():
                logger.info(f"name: {name}, module id: {id(module)}")

            group_index_list = resnet_groups_dream[dnn][nworkers][group_num][name]
            # group_index = resnet_groups[group_num][name]
            for param in module.parameters():
                p_tmp = param.expand_as(param)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(_make_hook(module, param, group_index_list, gap_iters=group_num, name=name, layer_index=layer_index))

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
        #logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
            
        for i in range(iters_per_epoch//nsteps_update):
            #_buffer_params = {}
            global_iters += 1
            result_dict = {}
            
            update_model_sgd_iters(trainer.net, i)
            s = time.time()
            optimizer.zero_grad()
            
            # with record_function("Train_models"):
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            
            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
                
            end_time = time.time()

            synchronize_all_reduced_models()
            wait_time = time.time() - end_time
            wait_time_acc += wait_time
                # logger.info(f'Global iteration: {global_iters} wait time: {wait_time} total wait time: {wait_time_acc}')
                
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            backward_time_acc += trainer.backwardtime_tmp
            
            logger.info(f'Global iteration: {global_iters} backward time: {trainer.backwardtime_tmp} train time: {train_time} \n wait time: {wait_time} total wait time: {wait_time_acc}')
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "total wait time": wait_time_acc , "total backward time":backward_time_acc, 
                        "total train time": train_time_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
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

    parser.add_argument('--model_dir', type=str, default='./model', help='')
    parser.add_argument('--load_pretrain', type=str, default='False', help='')

    parser.add_argument('--interface', default='eno0', help='Network interface, choosing from eno0-1G, ens5f0-10G')
    parser.add_argument('--alg', type=str,default='localsgd',help='Algorithms including desync, sgd, localsgd, layerwise.')
    parser.add_argument('--local_rank', type=int, default=0,help='local rank for distributed training')
    parser.add_argument('--group_num',type=int, default='6', help='Number of iterations to achieve full synchronziation in full_pipe_Seq.')
    parser.add_argument('--config_name', type=str, default='', help='Model configurations.')
    parser.add_argument('--model_name_or_path', type=str,default='',help='Local model path for GPT or Bert.')


    # Check model divergence
    parser.add_argument('--check_param_diversity', type=str, default="False")
    parser.add_argument('--nsteps_param_diversity', type=int, default=5)

    
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

    parser.add_argument("--enable_wandb", type=str, default="False")
    args = parser.parse_args()
    arg_str2bool(args)
    batch_size = args.batch_size * args.nsteps_update
    momentum_correction = args.momentum_correction != 0
    prefix = args.alg + '-' + 'lr_decay_' + args.lr_decay

  
    beijing_tz = pytz.timezone('Asia/Shanghai')

    logdir = '%s' % (datetime.datetime.now(beijing_tz).strftime("%m-%d-%H:%M")) + '-' + prefix

    if (args.alg == 'sgd'):
        directory_path = os.path.join('./test/sgd', args.dnn)
    elif (args.alg == 'localsgd'):
        directory_path = os.path.join('./test/localsgd', args.dnn)
    elif(args.alg == 'pipe_sgd'):
        directory_path = os.path.join('./test/pipeline', args.dnn)
    elif(args.alg == 'pipe_seq_localsgd'):
        directory_path = os.path.join('./test/pipe_seq_localsgd', args.dnn)
    elif(args.alg == 'pipe_seq_localsgd_warmup'):
        directory_path = os.path.join('./test/pipe_seq_localsgd_warmup', args.dnn)
    elif(args.alg == 'full_pipe_seq'):
        directory_path = os.path.join('./test/testing', args.dnn)
    elif(args.alg == 'dream_ddp'):
        directory_path = os.path.join('./test/dream_ddp', args.dnn)
    elif(args.alg == 'time_measure'):
        directory_path = os.path.join('./test/time_measure',args.dnn)
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
        # os.environ['NCCL_DEBUG'] = 'INFO'
        # os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        # os.environ['NCCL_DEBUG'] = 'TRACE'
        
        os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
        if args.interface == 'eno0':
            os.environ['NCCL_SOCKET_IFNAME'] = 'eno0' #,ens5f0
        elif args.interface == 'ens5f0':
            os.environ['NCCL_SOCKET_IFNAME'] = 'ens5f0'
        os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
        
        #logger.info(f"NCCL_SOCKET_IFNAME is set to: {os.environ.get('NCCL_SOCKET_IFNAME')}")
        dist.init_process_group(backend='nccl', init_method='env://')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank()
        #logger.info(f'The rank is consistent {rank == args.local_rank}')
        #print("The Torch.distributed is initialized by rank: ", rank)
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = None #SummaryWriter(tb_runs)

    set_seed(3000)
    ExpTool.init(args, dist)    
    
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    if (args.alg == 'localsgd'):
        logger.info("Alg used: localsgd.")
        localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.lr_decay, 
             args.check_param_diversity, args.nsteps_param_diversity)
    elif (args.alg == 'sgd'):
        logger.info("Alg used: sgd.")
        ssgd(args.optimizer_name, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix, args.lr_decay, 
             args.check_param_diversity, args.nsteps_param_diversity)
    elif (args.alg == 'pipe_sgd'):
        logger.info("Alg used: pipelined seq.")
        ssgd_with_pipe(args.optimizer_name,  args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix, args.lr_decay)
    elif (args.alg == 'pipe_seq_localsgd'):
        logger.info("Alg used: pipe_seq_localsgd.")
        pipe_seq_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.lr_decay, 
             args.check_param_diversity, args.nsteps_param_diversity)
    elif (args.alg == 'pipe_seq_localsgd_warmup'):
        logger.info("Alg used: pipe_seq_localsgd_warmup.")
        pipe_seq_localsgd_warmup(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.lr_decay, 
             args.check_param_diversity, args.nsteps_param_diversity)
    elif (args.alg == 'full_pipe_seq'):
        logger.info("Alg used: test.")
        test(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.lr_decay, args.group_num, 
             args.check_param_diversity, args.nsteps_param_diversity)
    elif (args.alg == 'dream_ddp'):
        logger.info("Alg used: test.")
        dream_ddp(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.lr_decay, args.group_num, 
             args.check_param_diversity, args.nsteps_param_diversity)
    if (args.alg == 'time_measure'):
        logger.info("Alg used: localsgd.")
        localsgd_measure(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.lr_decay)
        
    
    ExpTool.finish(args)

    #local_sgd_with_dist(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)


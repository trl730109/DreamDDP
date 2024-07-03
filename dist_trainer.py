# -*- coding: utf-8 -*-
# coding:utf-8
from __future__ import print_function
import time
import datetime
import torch
import numpy as np
import argparse, os
import settings

import pytz
import logging
from multiprocessing import set_start_method
from collections import defaultdict

from transformers import BertConfig, GPT2Config, BertForSequenceClassification, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dl_trainer import DLTrainer, _support_datasets, _support_dnns
from llm_trainer import LLMTrainer, _support_datasets, _support_dnns
from dist_utils import *
import dist_optimizer as dist_optim

from tensorboardX import SummaryWriter
from compression import compressors
from profiling import benchmark
from mpi4py import MPI

from helpers.exp_path import ExpTool

comm = MPI.COMM_WORLD
writer = None

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

def ssgd(optimizer_name, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    #print("Assign the gpu ", (rank%nwpernode)+2, " to the rank ", rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer)
    
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
    # optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, add_noise = add_noise, gaussian_mu = gaussian_mu, gaussian_std = gaussian_std, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    # trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1
    global_iters = 0
    
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
            for param in trainer.net.parameters():
                if param.requires_grad:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()
            
            if dnn in ['lstm', 'lstmwt2']:
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
            
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
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()
        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def ssgd_with_pipe(optimizer_name, add_noise, gaussian_mu, gaussian_std, overlap_scalar, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    #torch.manual_seed(rank)
    #print("Assign the gpu ", (rank%nwpernode)+2, " to the rank ", rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer)
    
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
        
    optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, add_noise = add_noise, gaussian_mu = gaussian_mu, gaussian_std = gaussian_std, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1
    global_iters = 0
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
            # for param in trainer.net.parameters():
            #     if param.requires_grad:
            #         dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            #         param.grad.data /= dist.get_world_size()
            
            # if dnn in ['lstm', 'lstmwt2']:
            #     optimizer.synchronize()
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     optimizer.synchronize()
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
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
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()    
            
        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name)
    
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

    # if settings.ORIGINAL_HOROVOD:
    #     optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(), backward_passes_per_step=nsteps_update)
    # else:
    #     optimizer = hvd.DistributedOptimizer(trainer.optimizer, overlap=overlap, overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    # trainer.update_optimizer(optimizer)
    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1

    global_iters = 0
    train_time_acc = 0
    comm_time_acc = 0
    iteration_time_acc = 0
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        #logger.info(f'The updates counts for each epoch is {str(iters_per_epoch//nsteps_update)}')
        #logger.info(f'The updates counts for each epoch is {str(iters_per_epoch)}')
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
            if dnn in ['lstm', 'lstmwt2']:
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
                
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
                
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()
            
            if global_iters % nsteps_localsgd == nsteps_localsgd - 1:
                start = time.time()
                avg_pseudo_gradients = allreduce_model_weights(trainer.net, compressors[compressor](), density, strategy, overlap_scalar)
                corrected_avg_pseudo_gradients = {'.'.join(name.split('.')[:-1]): value for name, value in avg_pseudo_gradients}
                trainer.net.load_state_dict(dict(corrected_avg_pseudo_gradients))
                comm_time_acc += (time.time() - start)
            else:
                pass
            iteration_time_acc += (time.time() - s)
            ExpTool.record({"global_iters": global_iters, "iteration time": iteration_time_acc, "total train time": train_time_acc,
                        "total comm time": comm_time_acc})

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()


def pipe_seq_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, sync='sum'):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name)

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

    # def backward_hook_factory(begin_comm_iter, gap_iters, layer_index):
    #     def hook(__module, grad_input, grad_output):
    #         # 检查当前迭代ID是否在指定的通信步骤中
    #         # if iter_id in communication_steps:
    #         # if is_root():
    #         #     logger.info(f"In backward hook, Iter: {__module.sgd_iters}, layer_index:{layer_index}, __module: {type(__module)}")
    #         # if __module.sgd_iters % gap_iters == begin_comm_iter:
    #         if is_communicate(__module, gap_iters, begin_comm_iter):
    #             if is_root():
    #                 logger.info(f"Cur iter: {__module.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
    #                             f"layer_index:{layer_index}, __module: {type(__module)}")
    #             for __param in __module.parameters():
    #                 # tensor = __param.data
    #                 # handle = dist.all_reduce(tensor, op=dist.ReduceOp.AVG, async_op=True)
    #                 # handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    #                 handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
    #                 # tensor += 1.0
    #                 # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    #                 # tensor /= dist.get_world_size()
    #                 _handles[__param] = (handle, None, 1)
    #                 if is_root():
    #                     if __param.requires_grad:
    #                         logger.info(f"__param. has grad :{__param.grad.data.shape}, norm: {__param.grad.data.norm()}")

    #                 # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
    #                 # _handles[__param] = (handle, None, 1)
    #         else:
    #             # if is_root():
    #             #     logger.info(f"Cur iter: {__module.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, not communicated")
    #             pass

    #     return hook

    def is_communicate(__module, gap_iters, begin_comm_iter):
        return __module.sgd_iters % gap_iters == begin_comm_iter

    import copy

    # if sync == "sum":
    #     def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
    #         def hook(*ignore):
    #             if is_communicate(__param, gap_iters, begin_comm_iter):
    #                 if is_root():
    #                     logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
    #                                 f"layer:{name}/{layer_index}-th, __module: {type(__module)}")
    #                 # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
    #                 # # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
    #                 # _handles[__param] = (handle, None, 1)
    #                 buffer_param = copy.deepcopy(__param.data)
    #                 handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
    #                 # handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.AVG, async_op=True)
    #                 _handles[__param] = (handle, None, 1)
    #                 _buffer_params[__param] = buffer_param

    #         return hook
    # elif sync == 'avg':
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
    # elif sync == 'sync_avg':
    #     def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
    #         def hook(*ignore):
    #             if is_communicate(__param, gap_iters, begin_comm_iter):
    #                 if is_root():
    #                     logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
    #                                 f"layer:{name}/{layer_index}-th, __module: {type(__module)}")
    #                 # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
    #                 # # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
    #                 # _handles[__param] = (handle, None, 1)
                    
    #                 buffer_param = copy.deepcopy(__param.data)
    #                 #handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
    #                 dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=False)
    #                 # _handles[__param] = (handle, None, 1)
    #                 #_buffer_params[__param] = buffer_param

    #         return hook


    named_modules = dict(trainer.net.named_modules())
    # for name, module in named_modules.items():
    #     logger.info(f"name: {name}, module: {id(module)}")

    layer_per_iter = int(len(named_modules) / nsteps_localsgd) + 1
    logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(named_modules)} "
                f"\n layer_per_iter:{layer_per_iter}")
    # hook = backward_hook_factory(layer_index // layer_per_iter, gap_iters=layer_per_iter, layer_index=layer_index)
    # hook = backward_hook_factory(layer_index // layer_per_iter, gap_iters=nsteps_localsgd, layer_index=layer_index)
    # module.register_backward_hook(hook)
    # module.register_full_backward_hook(hook)
    # for layer_index, (name, module) in enumerate(named_modules.items()):
    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if is_root():
            logger.info(f"name: {name}, module id: {id(module)}")
        # logger.info(f"name: {name}, module id: {id(module)}")
        for param in module.parameters():
            p_tmp = param.expand_as(param)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(_make_hook(module, param, layer_index // layer_per_iter, gap_iters=nsteps_localsgd, name=name, layer_index=layer_index))
            grad_accs.append(grad_acc)

    # for grad_acc in grad_accs:
    #     logger.info(f"grad_acc: {id(grad_acc)}")


    def update_model_sgd_iters(model, sgd_iters):
        for module in model.modules():
            module.sgd_iters = sgd_iters
            for param in module.parameters():
                param.sgd_iters = sgd_iters


    def synchronize_all_reduced_models():

        for tensor, value in _handles.items():
            handle, ctx, density = value
            handle.wait()
            # tensor.data -= 1.0
            # tensor.data = tensor.data * dist.get_world_size()
            # tensor /= dist.get_world_size()
            # tensor = _buffer_params[tensor]
            if sync == 'sum':
                #logger.info(f'Tensor {tensor} divided tensor{}')
                _buffer_params[tensor] = _buffer_params[tensor] / dist.get_world_size()
                tensor = _buffer_params[tensor]
            # elif sync == 'avg':
            #     tensor = _buffer_params[tensor] 
            # tensor.data = tensor.data / dist.get_world_size()
            # tensor.data.set_(tensor.data / dist.get_world_size())

        _handles.clear()
        _buffer_params.clear()

    global_iters = 0
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
            
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            if dnn in ['lstm', 'lstmwt2']:
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            if (sync != 'sync_avg'):
                synchronize_all_reduced_models()
                # if(sync == 'sum'):
                #     for tensor, buffer in _buffer_params.items():
                #         tensor = _buffer_params[tensor]

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()


def test(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, sync='sum'):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name)

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
                handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
                _handles[__param] = (handle, None, 1)

        return hook

    named_modules = dict(trainer.net.named_modules())

    layer_per_iter = len(named_modules) // nsteps_localsgd
    fewer_iters = nsteps_localsgd - len(named_modules) % nsteps_localsgd
    division_index = layer_per_iter * fewer_iters

    logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(named_modules)} "
                f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if is_root():
            logger.info(f"name: {name}, module id: {id(module)}")
        # logger.info(f"name: {name}, module id: {id(module)}")
        for param in module.parameters():
            p_tmp = param.expand_as(param)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            if(layer_index < division_index):
                grad_acc.register_hook(_make_hook(module, param, layer_index // layer_per_iter, gap_iters=nsteps_localsgd, name=name, layer_index=layer_index))
            else:
                grad_acc.register_hook(_make_hook(module, param, fewer_iters + (layer_index-division_index) // (layer_per_iter+1), gap_iters=nsteps_localsgd, name=name, layer_index=layer_index))
            grad_accs.append(grad_acc)

    def update_model_sgd_iters(model, sgd_iters):
        for module in model.modules():
            module.sgd_iters = sgd_iters
            for param in module.parameters():
                param.sgd_iters = sgd_iters

    def synchronize_all_reduced_models():

        for tensor, value in _handles.items():
            handle, ctx, density = value
            handle.wait()
            if sync == 'sum':
                #logger.info(f'Tensor {tensor} divided tensor{}')
                _buffer_params[tensor] = _buffer_params[tensor] / dist.get_world_size()
                tensor = _buffer_params[tensor]

        _handles.clear()
        _buffer_params.clear()

    global_iters = 0
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
            
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            if dnn in ['lstm', 'lstmwt2']:
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            if (sync != 'sync_avg'):
                synchronize_all_reduced_models()

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def transformer_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, max_epochs, nwpernode, nsteps_update, tokenizer_name=None, nsteps_localsgd=20, model_dir=None):
    assert nsteps_localsgd > 1
    set_seed(3000)
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    times = []
    trainer = LLMTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=None, num_steps=num_steps, tb_writer=writer,optimizer_name="Adam")
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    iters_per_epoch = trainer.num_batches_per_epoch
    
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')
    
    global_iters = 0
    train_time_acc = 0
    comm_time_acc = 0
    iteration_time_acc = 0
    
    for epoch in range(max_epochs):
        trainer.net.train()
        trainer.train_sampler.set_epoch(epoch)
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0  
        
        for j in range(iters_per_epoch):
            s = time.time()
            trainer.zero_grad()
            trainer.train(1)
            
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time() - s
            times.append(train_time)
            train_time_acc += train_time
            
            display = 40
            if j % display == 0 and j > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
                
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()
            
            if global_iters % nsteps_localsgd == nsteps_localsgd - 1:
                start = time.time()
                for param in trainer.net.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
                comm_time_acc += (time.time() - start)
            else:
                pass
            iteration_time_acc += (time.time() - s)
            ExpTool.record({"global_iters": global_iters, "iteration time": iteration_time_acc, "total train time": train_time_acc,
                        "total comm time": comm_time_acc})
            global_iters += 1
                
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def transformer_seq_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, sync='sum'):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = LLMTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name)

    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    
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

    def is_communicate(__module, gap_iters, begin_comm_iter):
        return __module.sgd_iters % gap_iters == begin_comm_iter

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

    named_modules = dict(trainer.net.named_modules())

    layer_per_iter = int(len(named_modules) / nsteps_localsgd) + 1
    logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(named_modules)} "
                f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if is_root():
            logger.info(f"name: {name}, module id: {id(module)}")
        # logger.info(f"name: {name}, module id: {id(module)}")
        for param in module.parameters():
            p_tmp = param.expand_as(param)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(_make_hook(module, param, layer_index // layer_per_iter, gap_iters=nsteps_localsgd, name=name, layer_index=layer_index))
            grad_accs.append(grad_acc)

    # for grad_acc in grad_accs:
    #     logger.info(f"grad_acc: {id(grad_acc)}")


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
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        
        for i in range(iters_per_epoch//nsteps_update):
            #_buffer_params = {}
            global_iters += 1
            result_dict = {}
            
            update_model_sgd_iters(trainer.net, i)
            s = time.time()
            optimizer.zero_grad()
            
            for j in range(nsteps_update):
                trainer.train(1)
                synchronize_all_reduced_models()

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
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
    set_seed(3000)
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
    #parser.add_argument('--gaussian_mu', type=float, default=0.0, help='Mean of the Gaussian Noise Mean.')
    #parser.add_argument('--gaussian_std', type=float, default=0.01, help='Std of the Gaussian Noise std.')
    #parser.add_argument('--add_noise', type=str, default='false', help='Whether to add noise to the averaged gradients.')
    parser.add_argument('--alg', type=str,default='localsgd',help='Algorithms including desync, sgd, localsgd, layerwise.')
    parser.add_argument('--local_rank', type=int, default=0,help='local rank for distributed training')
    parser.add_argument('--sync',type=str,default='sum',help='synchronization ways, sum or avg')
    
    parser.add_argument('--config_name', type=str, default='', help='Model configurations.')
    parser.add_argument('--model_name_or_path', type=str,default='',help='Local model path for GPT or Bert.')

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
    prefix = args.alg
    # prefix = prefix + '-' + args.optimizer_name + '-' + args.strategy
    # if args.add_noise:
    #     prefix = prefix + '-' + 'mu_' + str(args.gaussian_mu) + '-' + "std_" + str(args.gaussian_std)
    # if args.density < 1:
    #     if (args.strategy == 'overlap'):
    #         prefix = '-' + 'scalar-' + str(args.overlap_scalar) + '-' + 'comp-' + args.compressor + '-' + prefix
    #     else:
    #         prefix = '-' + 'comp-' + args.compressor + '-' + prefix
    #     if momentum_correction:
    #         prefix = 'mc-'+ prefix
  
    beijing_tz = pytz.timezone('Asia/Shanghai')

    logdir = '%s' % (datetime.datetime.now(beijing_tz).strftime("%m-%d-%H:%M")) + '-' + prefix

    if (args.alg == 'sgd'):
        directory_path = os.path.join('./test/sgd', args.dnn)
    elif (args.alg == 'localsgd'):
        directory_path = os.path.join('./test/localsgd', args.dnn)
    elif (args.alg == 'desync'):
        directory_path = os.path.join('./test/desync', args.dnn)
    elif(args.alg == 'layerwise'):
        directory_path = os.path.join('./test/layerwise', args.dnn)
    elif(args.alg == 'seq'):
        directory_path = os.path.join('./test/sequential', args.dnn)
    elif(args.alg == 'pipe'):
        directory_path = os.path.join('./test/pipeline', args.dnn)
    elif(args.alg == 'pipe_seq_localsgd'):
        directory_path = os.path.join('./test/pipe_seq_localsgd', args.dnn, args.sync)
    elif(args.alg == 'test'):
        directory_path = os.path.join('./test/testing', args.dnn, args.sync)
    elif(args.alg == 'transformer'):
        directory_path = os.path.join('./test/transformers', args.dnn)
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
        dist.init_process_group(backend='nccl', init_method='env://')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank()
        logger.info(f'The rank is consistent {rank == args.local_rank}')
        print("The Torch.distributed is initialized by rank: ", rank)
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = None #SummaryWriter(tb_runs)
        
    ExpTool.init(args, dist)    
    
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    if (args.alg == 'localsgd'):
        logger.info("Alg used: localsgd.")
        localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)

    elif (args.alg == 'sgd'):
        logger.info("Alg used: sgd.")
        ssgd(args.optimizer_name, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix)
    # elif (args.alg == 'seq'):
    #     logger.info("Alg used: seq.")
    #     seq_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)
    elif (args.alg == 'pipe'):
        logger.info("Alg used: pipelined seq.")
        ssgd_with_pipe(args.optimizer_name, args.add_noise, args.gaussian_mu, args.gaussian_std, args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix)
    elif (args.alg == 'pipe_seq_localsgd'):
        logger.info("Alg used: pipe_seq_localsgd.")
        pipe_seq_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.sync)
    elif (args.alg == 'test'):
        logger.info("Alg used: test.")
        test(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.sync)
    elif (args.alg == 'transformer'):
        logger.info("Alg used: transformer training.")
        train_transformer(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.max_epochs, args.nwpernode, args.nsteps_update, tokenizer_name=None, nsteps_localsgd=args.nsteps_localsgd, model_dir=args.model_name_or_path)
        
    ExpTool.finish(args)

    #local_sgd_with_dist(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)

# -*- coding: utf-8 -*-
# coding:utf-8
from __future__ import print_function
import time
import datetime
import torch
import numpy as np
import argparse, os
import settings

import pytz
import logging
from multiprocessing import set_start_method
from collections import defaultdict

from transformers import BertConfig, GPT2Config, BertForSequenceClassification, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dl_trainer import DLTrainer, _support_datasets, _support_dnns
from llm_trainer import LLMTrainer, _support_datasets, _support_dnns
from dist_utils import *
import dist_optimizer as dist_optim

from tensorboardX import SummaryWriter
from compression import compressors
from profiling import benchmark
from mpi4py import MPI

from helpers.exp_path import ExpTool

comm = MPI.COMM_WORLD
writer = None

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

def ssgd(optimizer_name, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    #print("Assign the gpu ", (rank%nwpernode)+2, " to the rank ", rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer)
    
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
    # optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, add_noise = add_noise, gaussian_mu = gaussian_mu, gaussian_std = gaussian_std, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    # trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1
    global_iters = 0
    
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
            for param in trainer.net.parameters():
                if param.requires_grad:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()
            
            if dnn in ['lstm', 'lstmwt2']:
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
            
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
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()
        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def ssgd_with_pipe(optimizer_name, add_noise, gaussian_mu, gaussian_std, overlap_scalar, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    #torch.manual_seed(rank)
    #print("Assign the gpu ", (rank%nwpernode)+2, " to the rank ", rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer)
    
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
        
    optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, add_noise = add_noise, gaussian_mu = gaussian_mu, gaussian_std = gaussian_std, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1
    global_iters = 0
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
            # for param in trainer.net.parameters():
            #     if param.requires_grad:
            #         dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            #         param.grad.data /= dist.get_world_size()
            
            # if dnn in ['lstm', 'lstmwt2']:
            #     optimizer.synchronize()
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     optimizer.synchronize()
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
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
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()    
            
        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name)
    
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

    # if settings.ORIGINAL_HOROVOD:
    #     optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(), backward_passes_per_step=nsteps_update)
    # else:
    #     optimizer = hvd.DistributedOptimizer(trainer.optimizer, overlap=overlap, overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    # trainer.update_optimizer(optimizer)
    optimizer = trainer.optimizer
    iters_per_epoch = trainer.num_batches_per_epoch
    #max_epochs=0

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1

    global_iters = 0
    train_time_acc = 0
    comm_time_acc = 0
    iteration_time_acc = 0
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        #logger.info(f'The updates counts for each epoch is {str(iters_per_epoch//nsteps_update)}')
        #logger.info(f'The updates counts for each epoch is {str(iters_per_epoch)}')
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
            if dnn in ['lstm', 'lstmwt2']:
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
                
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
                
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()
            
            if global_iters % nsteps_localsgd == nsteps_localsgd - 1:
                start = time.time()
                avg_pseudo_gradients = allreduce_model_weights(trainer.net, compressors[compressor](), density, strategy, overlap_scalar)
                corrected_avg_pseudo_gradients = {'.'.join(name.split('.')[:-1]): value for name, value in avg_pseudo_gradients}
                trainer.net.load_state_dict(dict(corrected_avg_pseudo_gradients))
                comm_time_acc += (time.time() - start)
            else:
                pass
            iteration_time_acc += (time.time() - s)
            ExpTool.record({"global_iters": global_iters, "iteration time": iteration_time_acc, "total train time": train_time_acc,
                        "total comm time": comm_time_acc})

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()


def pipe_seq_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, sync='sum'):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name)

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

    # def backward_hook_factory(begin_comm_iter, gap_iters, layer_index):
    #     def hook(__module, grad_input, grad_output):
    #         # 检查当前迭代ID是否在指定的通信步骤中
    #         # if iter_id in communication_steps:
    #         # if is_root():
    #         #     logger.info(f"In backward hook, Iter: {__module.sgd_iters}, layer_index:{layer_index}, __module: {type(__module)}")
    #         # if __module.sgd_iters % gap_iters == begin_comm_iter:
    #         if is_communicate(__module, gap_iters, begin_comm_iter):
    #             if is_root():
    #                 logger.info(f"Cur iter: {__module.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
    #                             f"layer_index:{layer_index}, __module: {type(__module)}")
    #             for __param in __module.parameters():
    #                 # tensor = __param.data
    #                 # handle = dist.all_reduce(tensor, op=dist.ReduceOp.AVG, async_op=True)
    #                 # handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    #                 handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
    #                 # tensor += 1.0
    #                 # dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    #                 # tensor /= dist.get_world_size()
    #                 _handles[__param] = (handle, None, 1)
    #                 if is_root():
    #                     if __param.requires_grad:
    #                         logger.info(f"__param. has grad :{__param.grad.data.shape}, norm: {__param.grad.data.norm()}")

    #                 # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
    #                 # _handles[__param] = (handle, None, 1)
    #         else:
    #             # if is_root():
    #             #     logger.info(f"Cur iter: {__module.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, not communicated")
    #             pass

    #     return hook

    def is_communicate(__module, gap_iters, begin_comm_iter):
        return __module.sgd_iters % gap_iters == begin_comm_iter

    import copy

    # if sync == "sum":
    #     def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
    #         def hook(*ignore):
    #             if is_communicate(__param, gap_iters, begin_comm_iter):
    #                 if is_root():
    #                     logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
    #                                 f"layer:{name}/{layer_index}-th, __module: {type(__module)}")
    #                 # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
    #                 # # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
    #                 # _handles[__param] = (handle, None, 1)
    #                 buffer_param = copy.deepcopy(__param.data)
    #                 handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
    #                 # handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.AVG, async_op=True)
    #                 _handles[__param] = (handle, None, 1)
    #                 _buffer_params[__param] = buffer_param

    #         return hook
    # elif sync == 'avg':
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
    # elif sync == 'sync_avg':
    #     def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
    #         def hook(*ignore):
    #             if is_communicate(__param, gap_iters, begin_comm_iter):
    #                 if is_root():
    #                     logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
    #                                 f"layer:{name}/{layer_index}-th, __module: {type(__module)}")
    #                 # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
    #                 # # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
    #                 # _handles[__param] = (handle, None, 1)
                    
    #                 buffer_param = copy.deepcopy(__param.data)
    #                 #handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
    #                 dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=False)
    #                 # _handles[__param] = (handle, None, 1)
    #                 #_buffer_params[__param] = buffer_param

    #         return hook


    named_modules = dict(trainer.net.named_modules())
    # for name, module in named_modules.items():
    #     logger.info(f"name: {name}, module: {id(module)}")

    layer_per_iter = int(len(named_modules) / nsteps_localsgd) + 1
    logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(named_modules)} "
                f"\n layer_per_iter:{layer_per_iter}")
    # hook = backward_hook_factory(layer_index // layer_per_iter, gap_iters=layer_per_iter, layer_index=layer_index)
    # hook = backward_hook_factory(layer_index // layer_per_iter, gap_iters=nsteps_localsgd, layer_index=layer_index)
    # module.register_backward_hook(hook)
    # module.register_full_backward_hook(hook)
    # for layer_index, (name, module) in enumerate(named_modules.items()):
    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if is_root():
            logger.info(f"name: {name}, module id: {id(module)}")
        # logger.info(f"name: {name}, module id: {id(module)}")
        for param in module.parameters():
            p_tmp = param.expand_as(param)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(_make_hook(module, param, layer_index // layer_per_iter, gap_iters=nsteps_localsgd, name=name, layer_index=layer_index))
            grad_accs.append(grad_acc)

    # for grad_acc in grad_accs:
    #     logger.info(f"grad_acc: {id(grad_acc)}")


    def update_model_sgd_iters(model, sgd_iters):
        for module in model.modules():
            module.sgd_iters = sgd_iters
            for param in module.parameters():
                param.sgd_iters = sgd_iters


    def synchronize_all_reduced_models():

        for tensor, value in _handles.items():
            handle, ctx, density = value
            handle.wait()
            # tensor.data -= 1.0
            # tensor.data = tensor.data * dist.get_world_size()
            # tensor /= dist.get_world_size()
            # tensor = _buffer_params[tensor]
            if sync == 'sum':
                #logger.info(f'Tensor {tensor} divided tensor{}')
                _buffer_params[tensor] = _buffer_params[tensor] / dist.get_world_size()
                tensor = _buffer_params[tensor]
            # elif sync == 'avg':
            #     tensor = _buffer_params[tensor] 
            # tensor.data = tensor.data / dist.get_world_size()
            # tensor.data.set_(tensor.data / dist.get_world_size())

        _handles.clear()
        _buffer_params.clear()

    global_iters = 0
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
            
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            if dnn in ['lstm', 'lstmwt2']:
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            if (sync != 'sync_avg'):
                synchronize_all_reduced_models()
                # if(sync == 'sum'):
                #     for tensor, buffer in _buffer_params.items():
                #         tensor = _buffer_params[tensor]

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()


def test(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, sync='sum'):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name)

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
                handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
                _handles[__param] = (handle, None, 1)

        return hook

    named_modules = dict(trainer.net.named_modules())

    layer_per_iter = len(named_modules) // nsteps_localsgd
    fewer_iters = nsteps_localsgd - len(named_modules) % nsteps_localsgd
    division_index = layer_per_iter * fewer_iters

    logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(named_modules)} "
                f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if is_root():
            logger.info(f"name: {name}, module id: {id(module)}")
        # logger.info(f"name: {name}, module id: {id(module)}")
        for param in module.parameters():
            p_tmp = param.expand_as(param)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            if(layer_index < division_index):
                grad_acc.register_hook(_make_hook(module, param, layer_index // layer_per_iter, gap_iters=nsteps_localsgd, name=name, layer_index=layer_index))
            else:
                grad_acc.register_hook(_make_hook(module, param, fewer_iters + (layer_index-division_index) // (layer_per_iter+1), gap_iters=nsteps_localsgd, name=name, layer_index=layer_index))
            grad_accs.append(grad_acc)

    def update_model_sgd_iters(model, sgd_iters):
        for module in model.modules():
            module.sgd_iters = sgd_iters
            for param in module.parameters():
                param.sgd_iters = sgd_iters

    def synchronize_all_reduced_models():

        for tensor, value in _handles.items():
            handle, ctx, density = value
            handle.wait()
            if sync == 'sum':
                #logger.info(f'Tensor {tensor} divided tensor{}')
                _buffer_params[tensor] = _buffer_params[tensor] / dist.get_world_size()
                tensor = _buffer_params[tensor]

        _handles.clear()
        _buffer_params.clear()

    global_iters = 0
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
            
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            if dnn in ['lstm', 'lstmwt2']:
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            if (sync != 'sync_avg'):
                synchronize_all_reduced_models()

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()  

        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def transformer_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, max_epochs, nwpernode, nsteps_update, tokenizer_name=None, nsteps_localsgd=20, model_dir=None):
    assert nsteps_localsgd > 1
    set_seed(3000)
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    times = []
    trainer = LLMTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=None, num_steps=num_steps, tb_writer=writer,optimizer_name="Adam")
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    iters_per_epoch = trainer.num_batches_per_epoch
    
    logger.info('Broadcast parameters....')
    broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')
    
    global_iters = 0
    train_time_acc = 0
    comm_time_acc = 0
    iteration_time_acc = 0
    
    for epoch in range(max_epochs):
        trainer.net.train()
        trainer.train_sampler.set_epoch(epoch)
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0  
        
        for j in range(iters_per_epoch):
            s = time.time()
            trainer.zero_grad()
            trainer.train(1)
            
            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            train_time = time.time() - s
            times.append(train_time)
            train_time_acc += train_time
            
            display = 40
            if j % display == 0 and j > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
                
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            ExpTool.upload()
            
            if global_iters % nsteps_localsgd == nsteps_localsgd - 1:
                start = time.time()
                for param in trainer.net.parameters():
                    dist.all_reduce(param.data, op=dist.ReduceOp.AVG)
                comm_time_acc += (time.time() - start)
            else:
                pass
            iteration_time_acc += (time.time() - s)
            ExpTool.record({"global_iters": global_iters, "iteration time": iteration_time_acc, "total train time": train_time_acc,
                        "total comm time": comm_time_acc})
            global_iters += 1
                
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()

def transformer_seq_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1, sync='sum'):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = LLMTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,optimizer_name=name)

    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
    
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

    def is_communicate(__module, gap_iters, begin_comm_iter):
        return __module.sgd_iters % gap_iters == begin_comm_iter

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

    named_modules = dict(trainer.net.named_modules())

    layer_per_iter = int(len(named_modules) / nsteps_localsgd) + 1
    logger.info(f"nsteps_localsgd:{nsteps_localsgd} \n len(modules): {len(named_modules)} "
                f"\n layer_per_iter:{layer_per_iter}")

    grad_accs = []
    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
        if is_root():
            logger.info(f"name: {name}, module id: {id(module)}")
        # logger.info(f"name: {name}, module id: {id(module)}")
        for param in module.parameters():
            p_tmp = param.expand_as(param)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(_make_hook(module, param, layer_index // layer_per_iter, gap_iters=nsteps_localsgd, name=name, layer_index=layer_index))
            grad_accs.append(grad_acc)

    # for grad_acc in grad_accs:
    #     logger.info(f"grad_acc: {id(grad_acc)}")


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
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        
        for i in range(iters_per_epoch//nsteps_update):
            #_buffer_params = {}
            global_iters += 1
            result_dict = {}
            
            update_model_sgd_iters(trainer.net, i)
            s = time.time()
            optimizer.zero_grad()
            
            for j in range(nsteps_update):
                trainer.train(1)
                synchronize_all_reduced_models()

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
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
    set_seed(3000)
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
    #parser.add_argument('--gaussian_mu', type=float, default=0.0, help='Mean of the Gaussian Noise Mean.')
    #parser.add_argument('--gaussian_std', type=float, default=0.01, help='Std of the Gaussian Noise std.')
    #parser.add_argument('--add_noise', type=str, default='false', help='Whether to add noise to the averaged gradients.')
    parser.add_argument('--alg', type=str,default='localsgd',help='Algorithms including desync, sgd, localsgd, layerwise.')
    parser.add_argument('--local_rank', type=int, default=0,help='local rank for distributed training')
    parser.add_argument('--sync',type=str,default='sum',help='synchronization ways, sum or avg')
    
    parser.add_argument('--config_name', type=str, default='', help='Model configurations.')
    parser.add_argument('--model_name_or_path', type=str,default='',help='Local model path for GPT or Bert.')

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
    prefix = args.alg
    # prefix = prefix + '-' + args.optimizer_name + '-' + args.strategy
    # if args.add_noise:
    #     prefix = prefix + '-' + 'mu_' + str(args.gaussian_mu) + '-' + "std_" + str(args.gaussian_std)
    # if args.density < 1:
    #     if (args.strategy == 'overlap'):
    #         prefix = '-' + 'scalar-' + str(args.overlap_scalar) + '-' + 'comp-' + args.compressor + '-' + prefix
    #     else:
    #         prefix = '-' + 'comp-' + args.compressor + '-' + prefix
    #     if momentum_correction:
    #         prefix = 'mc-'+ prefix
  
    beijing_tz = pytz.timezone('Asia/Shanghai')

    logdir = '%s' % (datetime.datetime.now(beijing_tz).strftime("%m-%d-%H:%M")) + '-' + prefix

    if (args.alg == 'sgd'):
        directory_path = os.path.join('./test/sgd', args.dnn)
    elif (args.alg == 'localsgd'):
        directory_path = os.path.join('./test/localsgd', args.dnn)
    elif (args.alg == 'desync'):
        directory_path = os.path.join('./test/desync', args.dnn)
    elif(args.alg == 'layerwise'):
        directory_path = os.path.join('./test/layerwise', args.dnn)
    elif(args.alg == 'seq'):
        directory_path = os.path.join('./test/sequential', args.dnn)
    elif(args.alg == 'pipe'):
        directory_path = os.path.join('./test/pipeline', args.dnn)
    elif(args.alg == 'pipe_seq_localsgd'):
        directory_path = os.path.join('./test/pipe_seq_localsgd', args.dnn, args.sync)
    elif(args.alg == 'test'):
        directory_path = os.path.join('./test/testing', args.dnn, args.sync)
    elif(args.alg == 'transformer'):
        directory_path = os.path.join('./test/transformers', args.dnn)
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
        dist.init_process_group(backend='nccl', init_method='env://')
        args.local_rank = int(os.environ['LOCAL_RANK'])
        rank = dist.get_rank()
        logger.info(f'The rank is consistent {rank == args.local_rank}')
        print("The Torch.distributed is initialized by rank: ", rank)
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = None #SummaryWriter(tb_runs)
        
    ExpTool.init(args, dist)    
    
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    if (args.alg == 'localsgd'):
        logger.info("Alg used: localsgd.")
        localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)

    elif (args.alg == 'sgd'):
        logger.info("Alg used: sgd.")
        ssgd(args.optimizer_name, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix)
    # elif (args.alg == 'seq'):
    #     logger.info("Alg used: seq.")
    #     seq_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)
    elif (args.alg == 'pipe'):
        logger.info("Alg used: pipelined seq.")
        ssgd_with_pipe(args.optimizer_name, args.add_noise, args.gaussian_mu, args.gaussian_std, args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix)
    elif (args.alg == 'pipe_seq_localsgd'):
        logger.info("Alg used: pipe_seq_localsgd.")
        pipe_seq_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.sync)
    elif (args.alg == 'test'):
        logger.info("Alg used: test.")
        test(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd, args.sync)
    elif (args.alg == 'transformer'):
        logger.info("Alg used: transformer training.")
        train_transformer(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.max_epochs, args.nwpernode, args.nsteps_update, tokenizer_name=None, nsteps_localsgd=args.nsteps_localsgd, model_dir=args.model_name_or_path)
        
    ExpTool.finish(args)

    #local_sgd_with_dist(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)


# -*- coding: utf-8 -*-
# coding:utf-8
from __future__ import print_function
import time
import datetime
import torch
import numpy as np
import argparse, os
import settings
import utils
import pytz
import logging
from multiprocessing import set_start_method
from collections import defaultdict

import torch.distributed as dist
from dl_trainer import DLTrainer, _support_datasets, _support_dnns
from dist_utils import *
import dist_optimizer as dist_optim
# if settings.ORIGINAL_HOROVOD:
#     import horovod.torch as hvd
# else:
#     import hv_distributed_optimizer as hvd
#     from hv_distributed_optimizer import allreduce_model_weights,allgather_layers
#     os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
#     os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
from tensorboardX import SummaryWriter
from compression import compressors
from profiling import benchmark
from mpi4py import MPI
from utils import *
comm = MPI.COMM_WORLD
writer = None

def is_root():
    return dist.get_rank() == 0


from settings import logger, formatter

def ssgd_with_dist(optimizer_name, add_noise, gaussian_mu, gaussian_std, overlap_scalar, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    if compressor in ['randomksame', 'randomksameec']:
        torch.manual_seed(3000)
        torch.cuda.manual_seed_all(3000)
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
        
    optimizer = trainer.optimizer
    # optimizer = dist_optim.DistributedOptimizer(trainer.optimizer, add_noise = add_noise, gaussian_mu = gaussian_mu, gaussian_std = gaussian_std, strategy=strategy,overlap_scalar=overlap_scalar, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    # trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.num_batches_per_epoch

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 1 if iters_per_epoch > 40 else iters_per_epoch-1
    for epoch in range(max_epochs):
        hidden = None
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
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
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)

def ssgd_with_pipe(optimizer_name, add_noise, gaussian_mu, gaussian_std, overlap_scalar, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    if compressor in ['randomksame', 'randomksameec']:
        torch.manual_seed(3000)
        torch.cuda.manual_seed_all(3000)
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
    for epoch in range(max_epochs):
        hidden = None
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
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
            
            # if dnn in ['lstm', 'lstmwt2']:
            #     optimizer.synchronize()
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            # elif dnn == 'lstman4':
            #     optimizer.synchronize()
            #     torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
            
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)

def local_sgd_with_dist(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    if compressor in ['randomksame', 'randomksameec']:
        torch.manual_seed(3000)
        torch.cuda.manual_seed_all(3000)

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

    total_iters = 0
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None
        #logger.info(f'The updates counts for each epoch is {str(iters_per_epoch//nsteps_update)}')
        #logger.info(f'The updates counts for each epoch is {str(iters_per_epoch)}')
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
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
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []

            if total_iters % nsteps_localsgd == nsteps_localsgd - 1:
                avg_pseudo_gradients = allreduce_model_weights(trainer.net, compressors[compressor](), density, strategy, overlap_scalar)
                corrected_avg_pseudo_gradients = {'.'.join(name.split('.')[:-1]): value for name, value in avg_pseudo_gradients}
                trainer.net.load_state_dict(dict(corrected_avg_pseudo_gradients))
            else:
                pass

            total_iters += 1
        val_acc = trainer.test(epoch)

def seq_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    if compressor in ['randomksame', 'randomksameec']:
        torch.manual_seed(3000)
        torch.cuda.manual_seed_all(3000)

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

    total_iters = 0
    layer_names = []
    for name in trainer.net.state_dict().keys():
        layer_names.append(name)
    regrouped_layers = group_layers(layer_names)
    layer_per_iter = int(len(regrouped_layers) / nsteps_localsgd)
    
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None

        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
            comm_layer_list = []
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
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []

            for j in range(layer_per_iter):
                #layer = regrouped_layers[len(regrouped_layers) - j - layer_per_iter * (i % nsteps_localsgd) -1]
                layer = regrouped_layers[j + layer_per_iter * (i % nsteps_localsgd)]
                for k in range(len(layer)):
                    comm_layer_list.append(layer[k])

            #logger.info(f'The comm layers for this iteration are {comm_layer_list}')

            if len(comm_layer_list) != 0:
                params = trainer.net.state_dict()
                layerwise_params = allgather_layers(params, strategy, comm_layer_list)
                trainer.net.load_state_dict(dict(layerwise_params))
            else:
                pass

            total_iters += 1

        val_acc = trainer.test(epoch)
        if not settings.ORIGINAL_HOROVOD:
            trainer.train_epoch += 1






def pipe_seq_localsgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, overlap_scalar, threshold,name, gradient_path=None, momentum_correction=False, prefix=None, nsteps_localsgd=1):
    assert nsteps_localsgd > 1
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
    #print("The ssgd_with_horovod is called by rank: ", rank)
    if compressor in ['randomksame', 'randomksameec']:
        torch.manual_seed(3000)
        torch.cuda.manual_seed_all(3000)

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

    total_iters = 0
    layer_names = []
    for name in trainer.net.state_dict().keys():
        layer_names.append(name)
    regrouped_layers = group_layers(layer_names)
    layer_per_iter = int(len(regrouped_layers) / nsteps_localsgd)

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
    def _make_hook(__module, __param, begin_comm_iter, gap_iters, name, layer_index):
        def hook(*ignore):
            # assert not p.grad.requires_grad
            # name = _parameter_names.get(p)
            # d_p = p.grad.data
            #handle = allreduce_async_(tensor, average=True, name=allreduce_name)
            # logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
            #             f"layer_index:{layer_index}, __module: {type(__module)}")
            # if is_root():
            #     logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter},  "
            #                 f"layer:{name}/{layer_index}-th, __module: {type(__module)}")
            if is_communicate(__param, gap_iters, begin_comm_iter):
                if is_root():
                    logger.info(f"Cur iter: {__param.sgd_iters} gap_iters:{gap_iters} begin_comm_iter:{begin_comm_iter}, communicated successfully "
                                f"layer:{name}/{layer_index}-th, __module: {type(__module)}")
                # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.SUM, async_op=True)
                # # handle = dist.all_reduce(__param.data, op=dist.ReduceOp.AVG, async_op=True)
                # _handles[__param] = (handle, None, 1)

                buffer_param = copy.deepcopy(__param.data)
                handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.SUM, async_op=True)
                # handle = dist.all_reduce(buffer_param, op=dist.ReduceOp.AVG, async_op=True)
                _handles[__param] = (handle, None, 1)
                _buffer_params[__param] = buffer_param

                # if is_root():
                #     if __param.requires_grad:
                #         logger.info(f"__param. layer:{name}/{layer_index}-th has grad :{__param.grad.data.shape}, norm: {__param.grad.data.norm()}")
            else:
                # if is_root():
                #     logger.info(f"Cur iter: {__param.sgd_iters} communicated fail "
                #                 f"layer_index:{layer_index}")
                pass
        return hook


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
        # for layer_index, module in enumerate(modules):
        #     layer_index // layer_per_iter, gap_iters=nsteps_localsgd, layer_index=layer_index

        for tensor, value in _handles.items():
            handle, ctx, density = value
            handle.wait()
            # tensor.data -= 1.0
            # tensor.data = tensor.data * dist.get_world_size()
            # tensor /= dist.get_world_size()
            # tensor = _buffer_params[tensor]
            tensor = _buffer_params[tensor] / dist.get_world_size()
            # tensor.data = tensor.data / dist.get_world_size()
            # tensor.data.set_(tensor.data / dist.get_world_size())

        _handles.clear()


    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        hidden = None

        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch//nsteps_update):
            update_model_sgd_iters(trainer.net, i)
            comm_layer_list = []
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

            synchronize_all_reduced_models()

            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                # logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []

            #logger.info(f'The comm layers for this iteration are {comm_layer_list}')

            if len(comm_layer_list) != 0:
                params = trainer.net.state_dict()
                layerwise_params = allgather_layers(params, strategy, comm_layer_list)
                trainer.net.load_state_dict(dict(layerwise_params))
            else:
                pass

            total_iters += 1

        val_acc = trainer.test(epoch)
        if not settings.ORIGINAL_HOROVOD:
            trainer.train_epoch += 1
















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
    parser.add_argument('--gaussian_mu', type=float, default=0.0, help='Mean of the Gaussian Noise Mean.')
    parser.add_argument('--gaussian_std', type=float, default=0.01, help='Std of the Gaussian Noise std.')
    parser.add_argument('--add_noise', type=str, default='false', help='Whether to add noise to the averaged gradients.')
    parser.add_argument('--alg', type=str,default='localsgd',help='Algorithms including desync, sgd, localsgd, layerwise.')
    parser.add_argument('--local_rank', type=int, default=0,help='local rank for distributed training')
    
    args = parser.parse_args()
    # logger.info(torch.distributed.is_nccl_available()) # 判断nccl是否可用
    # logger.info(torch.distributed.is_mpi_available())  # 判断mpi是否可用
    # logger.info(torch.distributed.is_gloo_available()) # 判断gloo是否可用

    # logger.info(f'hihihi {torch.distributed.is_available()}')
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
        directory_path = os.path.join('./test/pipe_seq_localsgd', args.dnn)
    relative_path = os.path.join(directory_path, logdir)

    print(relative_path)

    gradient_relative_path = None 
    utils.create_path(relative_path)
    if settings.LOGGING_GRADIENTS:
        gradient_relative_path = '%s/gradients/%s'%(args.saved_dir, logdir)
        utils.create_path(gradient_relative_path)
    rank = 0
    #set_start_method('spawn')
    print("Start initializing the horovod")
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
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    if (args.alg == 'localsgd'):
        logger.info("Alg used: localsgd.")
        local_sgd_with_dist(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)

    elif (args.alg == 'sgd'):
        logger.info("Alg used: sgd.")
        ssgd_with_dist(args.optimizer_name, args.add_noise, args.gaussian_mu, args.gaussian_std, args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix)
    elif (args.alg == 'seq'):
        logger.info("Alg used: seq.")
        seq_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)
    elif (args.alg == 'pipe'):
        logger.info("Alg used: pipelined seq.")
        ssgd_with_pipe(args.optimizer_name, args.add_noise, args.gaussian_mu, args.gaussian_std, args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix)
    elif (args.alg == 'pipe_seq_localsgd'):
        logger.info("Alg used: pipe_seq_localsgd.")
        pipe_seq_localsgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)



    #local_sgd_with_dist(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)

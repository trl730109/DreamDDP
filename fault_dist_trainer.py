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
import math

from copy import deepcopy

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
# from utils import *

from helpers.exp_path import ExpTool

comm = MPI.COMM_WORLD
writer = None

def is_root():
    return dist.get_rank() == 0


def str2bool(v):
    if isinstance(v, bool):
        return v
    # if v.lower() in ('yes', 'true', 't', 'y', '1'):
    if isinstance(v, str) and v.lower() in ('true', 'True'):
        return True
    elif isinstance(v, str) and v.lower() in ('false', 'False'):
        return False
    else:
        return v

def set_seed(seed=3000):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
from settings import logger, formatter


def add_nose_to_param_grad(param, gaussian_mu, gaussian_std):
    shape = param.grad.data.size()
    # gaussian_mu, gaussian_std
    # gaussian_noise = torch.normal(mean=gaussian_mu, std=gaussian_std*gaussian_noise, size=shape, device=param.grad.data.device)
    # gaussian_noise = torch.normal(mean=gaussian_mu, std=torch.max(gaussian_std*param.grad.data.abs(), torch.tensor(0.0001)))
    gaussian_noise = torch.normal(mean=gaussian_mu, std=gaussian_std, size=shape, device=param.grad.data.device)
    param.grad.data += gaussian_noise

def check_model_diff(model1, model2):
    pass


def param_diversity(model):
    if isinstance(model, dict):
        avg_params = deepcopy(model)
    else:
        avg_params = deepcopy(model.state_dict())

    # for name, module in model.name_modules():
    # for name, param in model.name_parameters():
    for name, param in avg_params.items():
        if "weight" in name:
            dist.all_reduce(param, op=dist.ReduceOp.SUM)
            param = param.float() / dist.get_world_size()
            # dist.all_reduce(param, op=dist.ReduceOp.AVG)

    if is_root():
        named_diversitys = {}
        # total_diversity = 0.0
        total_diversitys = []
        if isinstance(model, dict):
            for name, param in model.items():
                if "weight" in name:
                    diff = (avg_params[name] - param.data)
                    if param.dtype == torch.long:
                        diff = diff.float()
                    # named_diversitys[f"diver/{name}"] = diff.norm() / math.sqrt(diff.numel())
                    named_diversitys[name] = diff.norm() / math.sqrt(diff.numel())
                    # named_diversitys[f"diver/{name}"] = diff.norm()
                    total_diversitys.append(named_diversitys[name].item())
            # return named_diversitys, total_diversity
            return named_diversitys, np.mean(total_diversitys)
        else:
            for name, param in model.state_dict().items():
                if "weight" in name:
                    diff = (avg_params[name] - param.data)
                    if param.dtype == torch.long:
                        diff = diff.float()
                    # named_diversitys[f"diver/{name}"] = diff.norm() / math.sqrt(diff.numel())
                    named_diversitys[name] = diff.norm() / math.sqrt(diff.numel())
                    # named_diversitys[f"diver/{name}"] = diff.norm()
                    total_diversitys.append(named_diversitys[name].item())
            # return named_diversitys, total_diversity
            return named_diversitys, np.mean(total_diversitys)
    else:
        return None, None


def get_grad_norm(model):
    if is_root():
        named_norms = {}
        # total_norm = 0.0
        total_norms = []
        if isinstance(model, dict):
            for name, param in model.items():
                if "weight" in name:
                    named_norms[name] = param.grad.data.norm() / math.sqrt(param.grad.data.numel())
                    total_norms.append(named_norms[name].item())
            return named_norms, np.mean(total_norms)
        else:
            # for name, param in model.state_dict().items():
            for name, param in model.named_parameters():
                if "weight" in name:
                    named_norms[name] = param.grad.data.norm() / math.sqrt(param.grad.data.numel())
                    total_norms.append(named_norms[name].item())
            return named_norms, np.mean(total_norms)
    else:
        return None, None
    



def record_param_diversity_with_period(model, global_iters, nsteps_param_diversity, check_param_diversity):
    if check_param_diversity and (global_iters % nsteps_param_diversity == 0):
        named_diversitys, total_diversity = param_diversity(model)
        if is_root():
            new_named_diversitys = {}
            for layer, diversity in named_diversitys.items():
                new_named_diversitys[f"diver/{layer}"] = diversity
            ExpTool.record(named_diversitys)
            ExpTool.record({"total_diversity": total_diversity})
            logger.info(f'Params have diversity: {total_diversity} !!!!!!!!.')



def allreduce_model_weights(model):
    if isinstance(model, dict):
        state_dict = model
    else:
        state_dict = model.state_dict()
    # params = []
    handles = []
    for name, p in state_dict.items():
        # t = type(p)
        # p = torch.Tensor([p])
        # params.append((name, p))
        # dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
        # handle = dist.all_reduce(p, op=dist.ReduceOp.SUM, async_op=True)
        # handle = dist.all_reduce(state_dict[name], op=dist.ReduceOp.SUM, async_op=True)
        # handle = dist.all_reduce(state_dict[name].data, op=dist.ReduceOp.AVG, async_op=True)
        # handles.append(handle)
        # dist.all_reduce(state_dict[name].data, op=dist.ReduceOp.AVG)
        dist.all_reduce(state_dict[name].data, op=dist.ReduceOp.SUM)
    # for handle in handles:
    #     handle.wait()
    for name, p in state_dict.items():
        state_dict[name] = state_dict[name] / dist.get_world_size()

    return state_dict


def ssgd_with_dist(optimizer_name, add_noise, gaussian_mu, gaussian_std, overlap_scalar, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None,
                   nsteps_param_sync=None, check_param_diversity=None, nsteps_param_diversity=None, param_sync=None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,
                        lr_decay='general')
    
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
    for epoch in range(max_epochs):
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
                    if (str2bool(add_noise)):
                        add_nose_to_param_grad(param, gaussian_mu, gaussian_std)

            if dnn in ['lstm', 'lstmwt2']:
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc

            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            # if check_param_diversity and (global_iters % nsteps_param_diversity == 0):
            #     named_diversitys, total_diversity = param_diversity(trainer.net)
            #     if is_root():
            #         ExpTool.record(named_diversitys)
            #         ExpTool.record({"total_diversity": total_diversity})
            #         logger.info(f'Params have diversity: {total_diversity} !!!!!!!!.')
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)

            ExpTool.upload()


        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
        val_acc = trainer.test(epoch)
        result_dict["val_acc"] = val_acc
        result_dict["train_epoch_loss"] = train_epoch_loss / (iters_per_epoch//nsteps_update)
        result_dict["train_epoch_acc"] = train_epoch_acc / (iters_per_epoch//nsteps_update)

        ExpTool.record(result_dict)
        ExpTool.record({"global_iters": global_iters, "epochs": epoch})
        ExpTool.upload()





def ssgd_with_param_sync(optimizer_name, add_noise, gaussian_mu, gaussian_std, overlap_scalar, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, strategy, threshold, gradient_path=None, momentum_correction=False, prefix=None,
                        nsteps_param_sync=None, check_param_diversity=None, nsteps_param_diversity=None, param_sync=None): 
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, optimizer_name=optimizer_name, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer,
                        lr_decay='general')
    
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

    init_nsteps_param_sync = nsteps_param_sync
    new_nsteps_param_sync = nsteps_param_sync

    times = []
    logger.info('max_epochs: %d', max_epochs)
    display = 10 if iters_per_epoch > 40 else iters_per_epoch-1
    global_iters = 0
    for epoch in range(max_epochs):
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
                    if (str2bool(add_noise)):
                        add_nose_to_param_grad(param, gaussian_mu, gaussian_std)

            named_gradnorms, total_gradnorm = get_grad_norm(trainer.net)
            if dnn in ['lstm', 'lstmwt2']:
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
            elif dnn == 'lstman4':
                optimizer.synchronize()
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)

            train_loss = trainer.loss
            train_acc = np.mean(trainer.train_acc_top1)
            train_epoch_loss += train_loss
            train_epoch_acc += train_acc

            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                samples_per_seconds = batch_size * nsteps_update / time_per_iter
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = samples_per_seconds
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss,
                        "train_acc": train_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            if (global_iters % new_nsteps_param_sync == 0):
                logger.info(f'Params averaged using Allreduce at specific iterations.')
                avg_params = allreduce_model_weights(trainer.net)
                trainer.net.load_state_dict(dict(avg_params))
                named_diversitys, total_diversity = param_diversity(trainer.net)
                layers = list(named_diversitys.keys())
                diversitys = list(named_diversitys.values())
                max_index = np.argmax(diversitys)
                max_diversity = diversitys[max_index]
                argmax_layer = layers[max_index]
                diverge_per_iter = max_diversity / new_nsteps_param_sync
                max_error_per_iter = diverge_per_iter / trainer.lr
                
                grad_norm = named_gradnorms[argmax_layer]
                est_tolerance_iters = (grad_norm /10) // max_error_per_iter

                # grad_norm / max_error_per_iter
                ExpTool.record({"max_error_per_iter": max_error_per_iter, "argmax_error_grad_norm": grad_norm,
                                "est_tolerance_iters": est_tolerance_iters, "total_gradnorm": total_gradnorm})
                if param_sync == "detect_base":
                    # for i in range(1, 100):
                    #     if grad_norm * 1/100 < max_error_per_iter * i:
                    #         new_nsteps_param_sync = i
                    #         break
                    new_nsteps_param_sync = min(50, est_tolerance_iters)
                if is_root():
                    logger.info(f'Params have diversity: {total_diversity} after sync params !!!!!!!!.')
            ExpTool.upload()


        logger.info(f'The current training epoch is {trainer.get_train_epoch()}')
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
    parser.add_argument('--sync',type=str,default='sum',help='synchronization ways, sum or avg')

    # Check model divergence
    parser.add_argument('--nsteps_param_sync', type=int, default=20)
    parser.add_argument('--check_param_diversity', type=str, default="True")
    parser.add_argument('--nsteps_param_diversity', type=int, default=5)
    parser.add_argument('--param_sync', type=str, default="fix")

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
    elif(args.alg == 'sgd_with_sync'):
        directory_path = os.path.join('./test/sgd_with_sync', args.dnn)

    elif(args.alg == 'pipe'):
        directory_path = os.path.join('./test/pipeline', args.dnn)
    elif(args.alg == 'pipe_seq_localsgd'):
        directory_path = os.path.join('./test/pipe_seq_localsgd', args.dnn, args.sync)
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

    set_seed(seed=3000+rank)

    ExpTool.init(args, dist)

    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    if (args.alg == 'localsgd'):
        logger.info("Alg used: localsgd.")
        # local_sgd_with_dist(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)
        pass
    elif (args.alg == 'sgd'):
        logger.info("Alg used: sgd.")
        ssgd_with_dist(args.optimizer_name, args.add_noise, args.gaussian_mu, args.gaussian_std, args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix,
                       args.nsteps_param_sync, args.check_param_diversity, args.nsteps_param_diversity, args.param_sync)
    # elif (args.alg == 'pipe'):
    #     logger.info("Alg used: pipelined seq.")
    #     ssgd_with_pipe(args.optimizer_name, args.add_noise, args.gaussian_mu, args.gaussian_std, args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix)
    elif (args.alg == 'sgd_with_sync'):
        logger.info("Alg used: pipe_seq_localsgd.")
        ssgd_with_param_sync(args.optimizer_name, args.add_noise, args.gaussian_mu, args.gaussian_std, args.overlap_scalar, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy, args.threshold, gradient_relative_path, momentum_correction, prefix,
                       args.nsteps_param_sync, args.check_param_diversity, args.nsteps_param_diversity, args.param_sync)

    ExpTool.finish(args)


    #local_sgd_with_dist(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.strategy,args.overlap_scalar, args.threshold,args.optimizer_name, gradient_relative_path, momentum_correction, prefix, args.nsteps_localsgd)

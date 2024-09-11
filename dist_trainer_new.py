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
import math
import pytz
import logging
from copy import deepcopy
from multiprocessing import set_start_method
from collections import defaultdict
from multiprocessing import set_start_method
from collections import defaultdict

from transformers import BertConfig, GPT2Config, BertForSequenceClassification, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity
from dl_trainer import DLTrainer, _support_datasets, _support_dnns, create_net
from llm_trainer import LLMTrainer, _support_datasets, _support_dnns, _llms
from dist_utils import *
import dist_optimizer as dist_optim

from tensorboardX import SummaryWriter
from compression import compressors
from profiling import benchmark
from mpi4py import MPI

from helpers.exp_path import ExpTool
import layer_group
from layer_group import resnet_groups, resnet_groups_dream
from settings import logger, formatter
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

def clip_grad(model, dnn, max_norm):
    if dnn in ['lstm', 'lstmwt2']:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    elif dnn == 'lstman4':
        torch.nn.utils.clip_grad_norm_(model.parameters(), 400)
    elif dnn in ["gpt2", "bert-base-uncased", "llama2-124M"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2.0) 

def param_diversity(model, avg_params=None):
    if avg_params is None:
        if isinstance(model, dict):
            avg_params = deepcopy(model)
        else:
            avg_params = deepcopy(model.state_dict())

        for name, param in avg_params.items():
            if "weight" in name:
                dist.all_reduce(avg_params[name], op=dist.ReduceOp.AVG)

    if is_root():
        named_diversitys = {}
        # total_diversity = 0.0
        total_diversitys = []
        if isinstance(model, dict):
            for name, param in model.items():
                if "weight" in name and ("bn" not in name ):
                    diff = (avg_params[name] - param.data)
                    if param.dtype == torch.long:
                        logging.info(f"!!!!!!!!!!!!!!!!name is type torch.long!!!!!!!!!!!!!!!!")
                        diff = diff.float()
                    named_diversitys[name] = diff.norm() / math.sqrt(diff.numel())
                    named_diversitys[name] = named_diversitys[name].item()
                    total_diversitys.append(named_diversitys[name])
            return named_diversitys, np.mean(total_diversitys)
        else:
            for name, param in model.state_dict().items():
                if "weight" in name and ("bn" not in name ):
                    diff = (avg_params[name] - param.data)
                    if param.dtype == torch.long:
                        logging.info(f"!!!!!!!!!!!!!!!!name is type torch.long!!!!!!!!!!!!!!!!")
                        diff = diff.float()
                    named_diversitys[name] = diff.norm() / math.sqrt(diff.numel())
                    named_diversitys[name] = named_diversitys[name].item()
                    total_diversitys.append(named_diversitys[name])
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
            
def train(alg, dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, 
         prefix=None, nsteps_localsgd=1, lr_decay=None, 
        check_param_diversity=None, nsteps_param_diversity=None, args = None):
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)
        
    selected_gpu = rank%nwpernode
    torch.cuda.set_device(selected_gpu)
    if rank != 0:
        pretrain = None
    if dnn in _llms:
        trainer = LLMTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, 
                             data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=None, num_steps=35, tb_writer=writer,optimizer_name=args.optimizer_name, lr_decay=lr_decay,
                             args=args)
    else:
        trainer = DLTrainer(rank, nworkers,localsgd=True, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, 
                            data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=35, tb_writer=writer,optimizer_name=args.optimizer_name,lr_decay=lr_decay,
                            args=args)
    
    init_epoch = (torch.ones(1) * trainer.get_train_epoch()).to(selected_gpu)
    init_iter = (torch.ones(1) * trainer.get_train_iter()).to(selected_gpu)
    dist.broadcast(init_epoch, src=0)
    dist.broadcast(init_iter, src=0)
    trainer.set_train_epoch(int(init_epoch.item()))
    trainer.set_train_iter(int(init_iter.item()))
 
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
    compressor = compressors[args.compressor]()
    def sparse_allgather(param, ratio):
        shape = param.shape
        tensor = param.view(-1)
        k, _, ctx, selected_values = compressor.compress(tensor, None, ratio=ratio)
        # print(f'Index dtype is {ctx.dtype} values is {ctx}')
        index_list = [torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device) for _ in range(dist.get_world_size())]
        value_list = [torch.zeros(selected_values.shape, dtype=param.dtype, device=param.device) for _ in range(dist.get_world_size())]
        dist.all_gather(index_list, ctx)
        dist.all_gather(value_list, selected_values)
        new_param = torch.zeros_like(param).view(-1)
        for i in range(len(index_list)):
            new_param += compressor.decompress_new(value_list[i], index_list[i], name=None, shape=shape)
        print(f'Percentage of non zero values is {torch.count_nonzero(new_param)/new_param.numel()}')
        new_param /= dist.get_world_size()
        param = new_param.view(shape)
        compressor.clear()
        return param
    
    for epoch in range(max_epochs):
        logger.info(f"Trainer using the {trainer.optimizer_name} optimizer.")
        trainer.net.train()
        trainer.train_sampler.set_epoch(epoch)
        hidden = None
        # if dnn in ['lstm', 'lstmwt2']:
        #     hidden = trainer.net.init_hidden()
        
        result_dict = {}
        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        train_epoch_ppl = 0.0
        backward_list = []
            
        for i in range(iters_per_epoch//nsteps_update):
            global_iters += 1
            result_dict = {}
            s = time.time()
            optimizer.zero_grad()
            
            # for j in range(nsteps_update):
            #     if dnn in ['lstm', 'lstmwt2']:
            #             _, hidden = trainer.train(1, hidden=hidden)
            #     else:
            trainer.train(1)
            # Communicate the gradients
            if args.alg == 'sgd':
                for param in trainer.net.parameters():
                    if param.requires_grad:
                        dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
            
            clip_grad(trainer.net, dnn, GPT2_MAX_GRAD_NORM)
            
            train_loss = trainer.loss
            train_epoch_loss += train_loss
            if dnn in _llms:
                train_ppl = trainer.ppl
                train_epoch_ppl += train_ppl
            else:
                train_acc = np.mean(trainer.train_acc_top1)
                train_epoch_acc += train_acc
                
            trainer.update_model()
            train_time = time.time()-s
            times.append(train_time)
            train_time_acc += train_time
            # backward_time_acc += trainer.backwardtime_tmp
            # backward_list.append(trainer.backwardtime_tmp)
   
            trainer.backwardtime_tmp = 0.0
            
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f samples/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
                result_dict["time_per_iter"] = time_per_iter
                result_dict["samples_per_seconds"] = batch_size * nsteps_update / time_per_iter
                
            ExpTool.record(result_dict)
            ExpTool.record({"global_iters": global_iters, "epochs": epoch, "train_loss": train_loss})
            if dnn in _llms:
                ExpTool.record({"train_ppl":train_ppl})
            else:
                ExpTool.record({"train_acc": train_acc})
            record_param_diversity_with_period(trainer.net, global_iters, nsteps_param_diversity, check_param_diversity)
            ExpTool.upload()
            if (args.alg == 'localsgd'):         
                if global_iters % nsteps_localsgd == nsteps_localsgd - 1:
                    for layer_index, (name, module) in enumerate(trainer.net.named_modules()):
                        if len(list(module.children())) == 0:  
                            for param in module.parameters():
                                dist.all_reduce(param.data, op=dist.ReduceOp.AVG, async_op=False)
                        else:
                            pass
                    # synchronize optimizer
                    if args.sync_momentum:
                        if args.optimizer_name == 'SGD':
                            for group in trainer.optimizer.param_groups:
                                for p in group['params']:
                                    if 'momentum_buffer' in trainer.optimizer.state[p]:
                                        dist.all_reduce(trainer.optimizer.state[p]['momentum_buffer'], op=dist.ReduceOp.AVG, async_op=False)
                        elif args.optimizer_name == 'Adam':
                            for group in trainer.optimizer.param_groups:
                                for p in group['params']:
                                    if 'exp_avg' in trainer.optimizer.state[p] and 'exp_avg_sq' in trainer.optimizer.state[p]:
                                        if args.density < 1:
                                            sparse_allgather(trainer.optimizer.state[p]['exp_avg'], args.density)
                                            sparse_allgather(trainer.optimizer.state[p]['exp_avg_sq'], args.density)
                                        else:
                                            dist.all_reduce(trainer.optimizer.state[p]['exp_avg'], op=dist.ReduceOp.AVG, async_op=False)
                                            dist.all_reduce(trainer.optimizer.state[p]['exp_avg_sq'], op=dist.ReduceOp.AVG, async_op=False)
                else:
                    pass
            
            ExpTool.record({"global_iters": global_iters, "iteration time": iteration_time_acc, "total train time": train_time_acc,
                        "total comm time": comm_time_acc, "avg backward time": (backward_time_acc / global_iters)})

        if dnn in _llms:
            val_ppl, test_loss = trainer.test(epoch)
            result_dict["val_ppl"] = val_ppl
            result_dict["test_loss"] = test_loss
            result_dict["train_epoch_ppl"] = train_epoch_ppl / (iters_per_epoch//nsteps_update)
        else:
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
    parser.add_argument('--sync_momentum', type=str, default="False")

    
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
    # elif(args.alg == 'seq'):
    #     directory_path = os.path.join('./test/sequential', args.dnn)
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
    elif(args.alg == 'transformer_localsgd'):
        directory_path = os.path.join('./test/transformers_localsgd', args.dnn)
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
        if(args.dnn == 'cifar10' or args.dnn == 'cifar100'):        
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

    train(args.alg, args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, prefix, 
          args.nsteps_localsgd, args.lr_decay, args.check_param_diversity, args.nsteps_param_diversity, args)
    ExpTool.finish(args)
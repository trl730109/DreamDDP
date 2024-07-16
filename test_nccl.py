# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import datetime
import torch
import numpy as np
import argparse
import os
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
# from llm_trainer import LLMTrainer, _support_datasets, _support_dnns
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

    parser.add_argument('--interface', default='eno0', help='Network interface, choosing from eno0-1G, ens5f0-10G')
    parser.add_argument('--alg', type=str,default='localsgd',help='Algorithms including desync, sgd, localsgd, layerwise.')
    parser.add_argument('--local_rank', type=int, default=0,help='local rank for distributed training')
    parser.add_argument('--group_num',type=int, default='6', help='Number of iterations to achieve full synchronziation in full_pipe_Seq.')
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
    elif(args.alg == 'full_pipe_seq'):
        directory_path = os.path.join('./test/testing', args.dnn)
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
        # dist.init_process_group(backend='nccl')
        # rank = dist.get_rank()
        # os.environ['NCCL_DEBUG'] = 'INFO'
        # os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
        #os.environ['NCCL_SOCKET_IFNAME'] = args.interface
        os.environ['NCCL_DEBUG'] = 'TRACE'
        os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand
        os.environ['NCCL_SOCKET_IFNAME'] = 'ens5f0' #,ens5f0
        os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
        logger.info(f"NCCL_SOCKET_IFNAME is set to: {os.environ.get('NCCL_SOCKET_IFNAME')}")
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
    rank = dist.get_rank()
    logger.info('the rank of current process: %d', rank)

    selected_gpu = rank % args.nwpernode
    torch.cuda.set_device(selected_gpu)
    tensor = torch.randn(11700000).cuda()
    print(tensor.numel())
    # Warm up
    for _ in range(10):
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    dist.barrier()
    # Timing the all_reduce operation
    start_time = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    dist.barrier()
    elapsed_time = time.time() - start_time

    # Print the time taken
    if rank == 0:
        print(f"Time to all_reduce a tensor: {elapsed_time} seconds")
    
    dist.destroy_process_group()
# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import torch
import numpy as np
import argparse, os
import settings
import utils
import logging
from multiprocessing import set_start_method

from dl_trainer import DLTrainer, _support_datasets, _support_dnns
if settings.ORIGINAL_HOROVOD:
    import horovod.torch as hvd
else:
    import hv_distributed_optimizer as hvd
    os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
    os.environ['HOROVOD_CACHE_CAPACITY'] = '0'
from tensorboardX import SummaryWriter
from compression import compressors
from profiling import benchmark
from mpi4py import MPI
comm = MPI.COMM_WORLD
writer = None

from settings import logger, formatter


def ssgd_with_horovod(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, nwpernode, pretrain, num_steps, compressor, density, threshold, gradient_path=None, momentum_correction=False, prefix=None):
    rank = hvd.rank()
    if compressor in ['randomksame', 'randomksameec']:
        torch.manual_seed(3000)
        torch.cuda.manual_seed_all(3000)
    #torch.manual_seed(rank)
    torch.cuda.set_device(rank%nwpernode)
    if rank != 0:
        pretrain = None
    trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix, pretrain=pretrain, num_steps=num_steps, tb_writer=writer)

    init_epoch = torch.ones(1) * trainer.get_train_epoch()
    init_iter = torch.ones(1) * trainer.get_train_iter()
    trainer.set_train_epoch(int(hvd.broadcast(init_epoch, root_rank=0)[0]))
    trainer.set_train_iter(int(hvd.broadcast(init_iter, root_rank=0)[0]))
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
    hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')


    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    if settings.ORIGINAL_HOROVOD:
        optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(), backward_passes_per_step=nsteps_update)
    else:
        optimizer = hvd.DistributedOptimizer(trainer.optimizer, named_parameters=trainer.net.named_parameters(), compression=compressors[compressor](), is_sparse=is_sparse, density=density, seq_layernames=seq_layernames, layerwise_times=layerwise_times, norm_clip=norm_clip, threshold=threshold, writer=writer, gradient_path=gradient_path, momentum_correction=momentum_correction)
    trainer.update_optimizer(optimizer)
    iters_per_epoch = trainer.num_batches_per_epoch #trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)
    #max_epochs=0

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
        if not settings.ORIGINAL_HOROVOD:
            optimizer.train_epoch += 1


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
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    momentum_correction = args.momentum_correction != 0
    prefix = settings.PREFIX
    if args.density < 1:
        prefix = 'comp-' + args.compressor + '-' + prefix
        if momentum_correction:
            prefix = 'mc-'+ prefix
    prefix = 'allreduce-%s-thres-%dkbytes' % (prefix, args.threshold/1024)
    logdir = '%s/%s-n%d-bs%d-lr%.4f-ns%d-ds%s' % (prefix, args.dnn, args.nworkers, batch_size, args.lr, args.nsteps_update, str(args.density)) 
    relative_path = './logs/%s'%logdir
    gradient_relative_path = None 
    utils.create_path(relative_path)
    if settings.LOGGING_GRADIENTS:
        gradient_relative_path = '%s/gradients/%s'%(args.saved_dir, logdir)
        utils.create_path(gradient_relative_path)
    rank = 0
    #set_start_method('spawn')
    if args.nworkers > 1:
        hvd.init()
        rank = hvd.rank()
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = None #SummaryWriter(tb_runs)
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)
    ssgd_with_horovod(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.nwpernode, args.pretrain, args.num_steps, args.compressor, args.density, args.threshold, gradient_relative_path, momentum_correction, prefix)

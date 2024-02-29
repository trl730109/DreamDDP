# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import torch
import numpy as np
import sys
import argparse, os
import settings
import utils
import logging
import distributed_optimizer as dopt
from mpi4py import MPI
comm = MPI.COMM_WORLD
comm.Set_errhandler(MPI.ERRORS_RETURN)

from dl_trainer import DLTrainer, _support_datasets, _support_dnns
from compression import compressors
from profiling import benchmark

from settings import logger, formatter
import horovod.torch as hvd
from tensorboardX import SummaryWriter

writer = None

relative_path = None


def robust_ssgd(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, compression=False, compressor='topk', nwpernode=1, sigma_scale=2.5, pretrain=None, density=0.01, prefix=None, gpu=1):
    global relative_path
    is_cuda = gpu > 0
    ngpus = 1 if is_cuda else 0
    if is_cuda:
        torch.cuda.set_device(dopt.rank()%nwpernode)
    rank = dopt.rank()
    if rank != 0:
        pretrain = None

    trainer = DLTrainer(rank, nworkers, dist=False, batch_size=batch_size, is_weak_scaling=True, ngpus=ngpus, data_dir=data_dir, dataset=dataset, dnn=dnn, lr=lr, nworkers=nworkers, prefix=prefix+'-ds%s'%str(density), pretrain=pretrain, tb_writer=writer)

    init_epoch = trainer.get_train_epoch()
    init_iter = trainer.get_train_iter()

    trainer.set_train_epoch(comm.bcast(init_epoch))
    trainer.set_train_iter(comm.bcast(init_iter))

    def _error_handler(new_num_workers, new_rank):
        logger.info('Error info catched by trainer')
        trainer.update_nworker(new_num_workers, new_rank)
    if settings.LAYERWISE and settings.ADAPTIVE:
        seq_layernames, layerwise_times = None, None #benchmark(trainer)
        #layerwise_times = comm.bcast(layerwise_times, root=0)
        #logger.info('[rank:%d]layerwise backward times: %s', rank, zip(seq_layernames, layerwise_times))
    else:
        seq_layernames, layerwise_times = None, None

    compressor = compressor if compression else 'none'
    compressor = compressors[compressor]
    is_sparse = compression

    logger.info('Broadcast parameters....')
    hvd.broadcast_parameters(trainer.net.state_dict(), root_rank=0)
    logger.info('Broadcast parameters finished....')
    norm_clip = None
    if dnn == 'lstm':
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    optimizer = dopt.DistributedOptimizer(trainer.optimizer, trainer.net.named_parameters(), compression=compressor, is_sparse=is_sparse, err_handler=_error_handler, layerwise_times=layerwise_times, sigma_scale=sigma_scale, density=density, norm_clip=norm_clip, writer=writer, seq_layernames=seq_layernames)

    trainer.update_optimizer(optimizer)

    iters_per_epoch = trainer.get_num_of_training_samples() / (nworkers * batch_size * nsteps_update)

    times = []
    NUM_OF_DISLAY = 40
    display = NUM_OF_DISLAY if iters_per_epoch > NUM_OF_DISLAY else iters_per_epoch-1
    logger.info('Start training ....')
    for epoch in range(max_epochs):
        hidden = None
        if dnn == 'lstm':
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch):
            s = time.time()
            optimizer.zero_grad()
            for j in range(nsteps_update):
                if j < nsteps_update - 1 and nsteps_update > 1:
                    optimizer.local = True
                else:
                    optimizer.local = False
                if dnn == 'lstm':
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            if dnn == 'lstm':
                #optimizer.synchronize()
                #torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 0.25)
                pass
            elif dnn == 'lstman4':
                #optimizer.synchronize()
                #torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), 400)
                pass
            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f, Speed: %f images/s, current density: %f', time_per_iter, batch_size * nsteps_update / time_per_iter, optimizer.get_current_density())
                times = []
        optimizer.add_train_epoch()
        # For comparison purpose ===>
        if settings.LOGGING_ASSUMPTION and rank == 0:
            fn = os.path.join(relative_path, 'topknorm-rank%d-epoch%d.npy' % (rank, epoch))
            np.save(fn, optimizer._allreducer._profiling_norms)
        if settings.LOGGING_ASSUMPTION:
            optimizer._allreducer._profiling_norms = {}
    optimizer.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AllReduce trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--nwpernode', type=int, default=1, help='Number of workers per node')
    parser.add_argument('--gpu', type=int, default=1, help='Use GPU by default')
    parser.add_argument('--compression', dest='compression', action='store_true')
    parser.add_argument('--compressor', type=str, default='topk', choices=compressors.keys(), help='Specify the compressors if \'compression\' is open')
    parser.add_argument('--sigma-scale', type=float, default=2.5, help='Maximum sigma scaler for sparsification')
    parser.add_argument('--density', type=float, default=0.01, help='Density for sparsification')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=_support_datasets, help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=settings.MAX_EPOCHS, help='Default maximum epochs to train')
    parser.add_argument('--pretrain', type=str, default=None, help='Specify the pretrain path')
    parser.set_defaults(compression=False)
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    prefix = settings.PREFIX
    if args.compression:
        prefix = 'comp-' + args.compressor + '-' + prefix
    logdir = 'allreduce-%s/%s-n%d-bs%d-lr%.4f-ns%d-sg%.2f-ds%s' % (prefix, args.dnn, args.nworkers, batch_size, args.lr, args.nsteps_update, args.sigma_scale, str(args.density))
    relative_path = './logs/%s'%logdir
    utils.create_path(relative_path)
    rank = 0
    rank = dopt.rank()
    hvd.init()
    if rank == 0:
        tb_runs = './runs/%s'%logdir
        writer = SummaryWriter(tb_runs)
    logfile = os.path.join(relative_path, settings.hostname+'-'+str(rank)+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)
    
    logger.info('Interpreter: %s', sys.version)
    robust_ssgd(args.dnn, args.dataset, args.data_dir, args.nworkers, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.compression, args.compressor, args.nwpernode, args.sigma_scale, args.pretrain, args.density, prefix, args.gpu)

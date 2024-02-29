# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import time
import torch
import logging
import utils
from mpi4py import MPI
import settings
from settings import logger, LAYERWISE, ADAPTIVE


class MESSAGE:
    STOP = 'STOP'
    RUNNING = 'RUNNING'

mpi_float16 = MPI.BYTE.Create_contiguous(2).Commit()
MPI._typedict['e'] = mpi_float16
MPI_TYPES = {
        np.float32: MPI.FLOAT,
        np.float16: mpi_float16
        }

if LAYERWISE:
    #THRESHOLD = 1024 #256*1024 #256*1024 #64*1024*1024
    #THRESHOLD = 16*8192 #64*1024*1024
    THRESHOLD = 640*1024*1024
else:
    THRESHOLD = 64*1024*1024
alpha = 1.e-6 # linear model of allreduce time t = alpha + beta * x


def topk_sparse_allreduce(comm, sparse_tensor, storage, indexes=None, dtype=np.float32):
    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.01)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy().astype(np.uint32)
        k = len(indexes)
        values = tensor#[indexes] 

    num_workers = comm.size
    if storage is not None and 'values_1d' in storage:
        values_1d = storage['values_1d']
        indexes_1d = storage['indexes_1d']
        result = storage['result']
    else:
        values_1d = np.zeros(k * num_workers, dtype=np.float32)
        indexes_1d = np.zeros(k * num_workers, dtype=np.uint32)
        result = np.zeros_like(tensor) 
        storage['values_1d'] = values_1d
        storage['indexes_1d'] = indexes_1d
        storage['result'] = result
        
    if dtype != np.float32:
        values_1d = values_1d.astype(dtype)


    if len(indexes) == 0:
        return result, None

    nnz = k
    comm.Allgather(values, values_1d[:num_workers*nnz])
    comm.Allgather(indexes, indexes_1d[:num_workers*nnz])
    #comm.Barrier()

    #for i in range(num_workers):
    #    index = indexes_1d[i*nnz:(i+1)*nnz]
    #    result[index] += values_1d[i*nnz:(i+1)*nnz]
    #return result, None
    return values_1d, indexes_1d, None #result, None


def topk(tensor, k):
    indexes = np.abs(tensor).argsort()[-k:][::-1]
    return indexes, tensor[indexes]

def gtopk_sparse_allreduce(comm, sparse_tensor, storage=None, indexes=None, dtype=np.float32):
    """
    0: 0(0) <- 1(1), 2(2) <- 3(3), 4(4) <- 5(5), 6(6) <- 7(7)
    1: 0(0) <- 2(1), 4(2) <- 6(3)
    2: 0(0) <- 4(1)
    0 -> 1
    0 -> 2, 1 -> 3
    0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7
    """
    num_workers = comm.size
    rank = comm.rank

    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.001)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy()
        k = len(indexes)
        values = tensor 
    original_indexes = indexes
    send_values = np.concatenate((indexes, values))
    send_values[0:k] = indexes.astype(np.uint32)
    send_values[k:2*k] = values.astype(np.float32)
    if storage is not None and 'result_v2' in storage:
        recv_values = storage['result_v2']
        if recv_values.size < k*2:
            recv_values = np.zeros_like(send_values)
            if storage:
                storage['result_v2'] = recv_values
        recv_values = recv_values[0:k*2]
    else:
        recv_values = np.zeros_like(send_values)
        if storage:
            storage['result_v2'] = recv_values

    num_round = int(np.log2(num_workers))
    local_rank = rank
    exist_workers = num_workers
    step = 1
    participate_ranks = range(0, num_workers, step)
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank+1]
                comm.Recv([recv_values, MPI.FLOAT], source=source)
                tmp_indexes = recv_values[0:k].astype(np.int)
                tmp_values = recv_values[k:2*k]

                cv, c1, c2 = np.intersect1d(indexes, tmp_indexes, assume_unique=False, return_indices=True)
                values[c1] += tmp_values[c2]
                tmp_values[c2] = 0.0

                tmp_c = np.concatenate((values, tmp_values))
                tmp_topki, tmp_topkv = utils.topk(tmp_c, k)
                first_array_indexes = tmp_topki[tmp_topki < k]
                second_array_indexes = tmp_topki[tmp_topki >= k]-k
                indexes = np.concatenate((indexes[first_array_indexes], tmp_indexes[second_array_indexes]))
                values = np.concatenate((values[first_array_indexes], tmp_values[second_array_indexes]))

                send_values = np.concatenate((indexes, values))
                send_values[0:k] = indexes.astype(np.uint32)
                send_values[k:2*k] = values.astype(np.float32)
            else:
                target = participate_ranks[local_rank-1]
                logger.debug('[round:%d], %d(%d)->%d(%d)', i, rank, local_rank, target, local_rank-1)
                comm.Send([send_values, MPI.FLOAT], dest=target)
        exist_workers /= 2
        step *= 2
        participate_ranks = range(0, num_workers, step)
        comm.Barrier()

    if rank == 0:
        send_values = np.concatenate((indexes, values))
        indexes = indexes.astype(np.uint32)
        values = values.astype(np.float32)
        send_values[0:k] = indexes
        send_values[k:2*k] = values
    else:
        send_values = recv_values[0:2*k]
    comm.Bcast(send_values, root=0)
    tensor.fill(0.)
    if rank != 0:
        tmp_indexes = send_values[0:k].astype(np.uint32)
        tmp_values = send_values[k:2*k].astype(np.float32)
        values = tmp_values
        indexes = tmp_indexes

    cv, c1, c2 = np.intersect1d(original_indexes, indexes, assume_unique=False, return_indices=True)
    included_indexes = c1
    return values, indexes, included_indexes # final selected values and indexes


def meangtopk_sparse_allreduce(comm, sparse_tensor, storage=None, indexes=None, dtype=np.float32):
    """
    0: 0(0) <- 1(1), 2(2) <- 3(3), 4(4) <- 5(5), 6(6) <- 7(7)
    1: 0(0) <- 2(1), 4(2) <- 6(3)
    2: 0(0) <- 4(1)
    0 -> 1
    0 -> 2, 1 -> 3
    0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7
    """
    mean_and_var = np.array([indexes.mean, indexes.var], dtype=np.float32)
    full_mean_and_var = np.zeros_like(mean_and_var)
    comm.Allreduce(mean_and_var, full_mean_and_var, op=MPI.SUM)

    num_workers = comm.size
    rank = comm.rank
    full_mean = float(full_mean_and_var[0])/num_workers
    full_var = float(full_mean_and_var[1])/num_workers

    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.001)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy()
        k = len(indexes)
        values = tensor 
    original_indexes = indexes
    send_values = np.concatenate((indexes, values))
    send_values[0:k] = indexes.astype(np.uint32)
    send_values[k:2*k] = values.astype(np.float32)
    if storage is not None and 'result_v2' in storage:
        recv_values = storage['result_v2']
        if recv_values.size < k*2:
            recv_values = np.zeros_like(send_values)
            if storage:
                storage['result_v2'] = recv_values
        recv_values = recv_values[0:k*2]
    else:
        recv_values = np.zeros_like(send_values)
        if storage:
            storage['result_v2'] = recv_values

    num_round = int(np.log2(num_workers))
    local_rank = rank
    exist_workers = num_workers
    step = 1
    participate_ranks = range(0, num_workers, step)
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank+1]
                comm.Recv([recv_values, MPI.FLOAT], source=source)
                tmp_indexes = recv_values[0:k].astype(np.int)
                tmp_values = recv_values[k:2*k]

                cv, c1, c2 = np.intersect1d(indexes, tmp_indexes, assume_unique=False, return_indices=True)
                values[c1] += tmp_values[c2]
                tmp_values[c2] = 0.0

                tmp_c = np.concatenate((values, tmp_values))
                tmp_topki, tmp_topkv = utils.topk(tmp_c, k)
                first_array_indexes = tmp_topki[tmp_topki < k]
                second_array_indexes = tmp_topki[tmp_topki >= k]-k
                indexes = np.concatenate((indexes[first_array_indexes], tmp_indexes[second_array_indexes]))
                values = np.concatenate((values[first_array_indexes], tmp_values[second_array_indexes]))

                send_values = np.concatenate((indexes, values))
                send_values[0:k] = indexes.astype(np.uint32)
                send_values[k:2*k] = values.astype(np.float32)
            else:
                target = participate_ranks[local_rank-1]
                logger.debug('[round:%d], %d(%d)->%d(%d)', i, rank, local_rank, target, local_rank-1)
                comm.Send([send_values, MPI.FLOAT], dest=target)
        exist_workers /= 2
        step *= 2
        participate_ranks = range(0, num_workers, step)
        comm.Barrier()

    if rank == 0:
        send_values = np.concatenate((indexes, values))
        indexes = indexes.astype(np.uint32)
        values = values.astype(np.float32)
        send_values[0:k] = indexes
        send_values[k:2*k] = values
    else:
        send_values = recv_values[0:2*k]
    comm.Bcast(send_values, root=0)
    tensor.fill(0.)
    if rank != 0:
        tmp_indexes = send_values[0:k].astype(np.uint32)
        tmp_values = send_values[k:2*k].astype(np.float32)
        values = tmp_values
        indexes = tmp_indexes

    cv, c1, c2 = np.intersect1d(original_indexes, indexes, assume_unique=False, return_indices=True)
    included_indexes = c1
    return values, indexes, included_indexes, full_mean, full_var


def gtopk2_sparse_allreduce(comm, sparse_tensor, storage=None, indexes=None, dtype=np.float32):
    """
    0: 0(0) <- 1(1), 2(2) <- 3(3), 4(4) <- 5(5), 6(6) <- 7(7)
    1: 0(0) <- 2(1), 4(2) <- 6(3)
    2: 0(0) <- 4(1)
    0 -> 1
    0 -> 2, 1 -> 3
    0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7
    """
    num_workers = comm.size
    rank = comm.rank

    tensor = sparse_tensor
    if indexes is None:
        k = int(tensor.size * 0.001)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy()
        k = len(indexes)
        values = tensor 
    original_indexes = indexes
    indexes_ranks = np.zeros_like(indexes).astype(np.uint32)
    indexes_ranks.fill(rank)
    #remote_indexes_ranks = np.ones_like(indexes_ranks).astype(np.uint32)
    send_values = np.concatenate((indexes, values, indexes_ranks))
    send_values[0:k] = indexes.astype(np.uint32)
    send_values[k:2*k] = values.astype(np.float32)
    send_values[2*k:3*k] = indexes_ranks.astype(np.uint32)
    if storage is not None and 'result_v2' in storage:
        recv_values = storage['result_v2']
        if recv_values.size < k*3:
            recv_values = np.zeros_like(send_values)
            if storage:
                storage['result_v2'] = recv_values
        recv_values = recv_values[0:k*3]
    else:
        recv_values = np.zeros_like(send_values)
        if storage:
            storage['result_v2'] = recv_values

    num_round = int(np.log2(num_workers))
    local_rank = rank
    exist_workers = num_workers
    step = 1
    participate_ranks = range(0, num_workers, step)
    for i in range(num_round):
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank+1]
                comm.Recv([recv_values, MPI.FLOAT], source=source)

                tmp_indexes = recv_values[0:k].astype(np.int)
                tmp_values = recv_values[k:2*k]
                tmp_indexes_ranks = recv_values[2*k:3*k].astype(np.uint32)

                cv, c1, c2 = np.intersect1d(indexes, tmp_indexes, assume_unique=False, return_indices=True)
                values[c1] += tmp_values[c2]
                tmp_values[c2] = 0.0 

                tmp_c = np.concatenate((values, tmp_values))
                tmp_topki, tmp_topkv = utils.topk(tmp_c, k)
                first_array_indexes = tmp_topki[tmp_topki < k]
                second_array_indexes = tmp_topki[tmp_topki >= k]-k
                indexes = np.concatenate((indexes[first_array_indexes], tmp_indexes[second_array_indexes]))
                values = np.concatenate((values[first_array_indexes], tmp_values[second_array_indexes]))

                indexes_ranks[second_array_indexes] = tmp_indexes_ranks[second_array_indexes]

                send_values = np.concatenate((indexes, values, indexes_ranks))
                send_values[0:k] = indexes.astype(np.uint32)
                send_values[k:2*k] = values.astype(np.float32)
                send_values[2*k:3*k] = indexes_ranks.astype(np.uint32)
            else:
                target = participate_ranks[local_rank-1]
                logger.debug('[round:%d], %d(%d)->%d(%d)', i, rank, local_rank, target, local_rank-1)
                comm.Send([send_values, MPI.FLOAT], dest=target)
        exist_workers /= 2
        step *= 2
        participate_ranks = range(0, num_workers, step)
        comm.Barrier()

    if rank == 0:
        send_values = np.concatenate((indexes, values, indexes_ranks))
        indexes = indexes.astype(np.uint32)
        values = values.astype(np.float32)
        indexes_ranks = indexes_ranks.astype(np.uint32)
        send_values[0:k] = indexes
        send_values[k:2*k] = values
        send_values[2*k:3*k] = indexes_ranks
    else:
        send_values = recv_values[0:3*k]
    comm.Bcast(send_values, root=0)
    if rank != 0:
        tmp_indexes = send_values[0:k].astype(np.uint32)
        tmp_values = send_values[k:2*k].astype(np.float32)
        indexes_ranks = send_values[2*k:3*k].astype(np.uint32)
        values = tmp_values
        indexes = tmp_indexes
    #logger.info('[rank:%d], k: %d, indexes_ranks: %s', rank, k, indexes_ranks)
    #comm.Barrier()
    #included_indexes = np.nonzero(indexes_ranks == rank)[0]
    #logger.info('[rank:%d], included_indexes: %s', rank, included_indexes)
    #comm.Barrier()
    #included_values = values[included_indexes]
    cv, c1, c2 = np.intersect1d(tensor, values, assume_unique=False, return_indices=True)
    #logger.info('[rank:%d], tensor: %s, selected: %s', rank, tensor, values[c2])
    #comm.Barrier()
    included_indexes = c1
    #logger.info('[rank:%d], final included_indexes: %s', rank, included_indexes)
    #comm.Barrier()
    return values, indexes, included_indexes # final selected values and indexes


def atopk_sparse_allreduce(comm, sparse_tensor, storage=None, indexes=None, dtype=np.float32):
    """
    0: 0(0) <- 1(1), 2(2) <- 3(3), 4(4) <- 5(5), 6(6) <- 7(7)
    1: 0(0) <- 2(1), 4(2) <- 6(3)
    2: 0(0) <- 4(1)
    0 -> 1
    0 -> 2, 1 -> 3
    0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7
    """
    num_workers = comm.size
    rank = comm.rank

    tensor = sparse_tensor.flatten()
    if indexes is None:
        k = int(tensor.size * 0.01)
        indexes, values = utils.topk(tensor, k)
    else:
        if not (type(indexes) is np.ndarray):
            indexes = indexes.cpu().numpy().astype(np.uint32)
        k = len(indexes)
        values = tensor[indexes] 
    #logger.info('original max_indexes: %d', np.max(indexes))
    original_indexes = indexes
    send_values = np.concatenate((indexes, values))#.astype(np.float32)
    send_values[0:k] = indexes.astype(np.uint32)
    send_values[k:] = values.astype(np.float32)
    if storage is not None and 'result_v2' in storage:
        recv_values = storage['result_v2']
        if recv_values.size < k*2:
            recv_values = np.zeros_like(send_values)
            if storage:
                storage['result_v2'] = recv_values
        recv_values = recv_values[0:k*2]
    else:
        recv_values = np.zeros_like(send_values)
        if storage:
            storage['result_v2'] = recv_values

    num_round = int(np.log2(num_workers))
    local_rank = rank
    exist_workers = num_workers
    step = 1
    participate_ranks = range(0, num_workers, step)
    for i in range(num_round):
            #logger.debug('[round:%d][rank:%d] send_values: %s', i, rank, send_values)
        if rank in participate_ranks:
            local_rank = participate_ranks.index(rank)
            logger.debug('[round:%d]pr: %s, step:%d, local_rank:%d, rank:%d', i, participate_ranks, step, local_rank, rank)
            if local_rank % 2 == 0:
                source = participate_ranks[local_rank+1]
                logger.debug('[round:%d], %d(%d)<-%d(%d)', i, rank, local_rank, source, local_rank+1)
                #logger.debug('[%d]recv k: %d', rank, recv_values.size)
                comm.Recv([recv_values, MPI.FLOAT], source=source)
                tmp_indexes = recv_values[0:k].astype(np.int)
                tmp_values = recv_values[k:]
                #logger.debug('[round:%d][rank:%d] recv values: %s', i, rank, recv_values)

                cv, c1, c2 = np.intersect1d(indexes, tmp_indexes, assume_unique=False, return_indices=True)
                #logger.debug('[round:%d][rank:%d] c1max: %d, c2max: %d, lenofarray:%d', i, rank, np.max(c1), np.max(c2), len(values))
                values[c1] += tmp_values[c2]

                tmp_c = np.concatenate((values, tmp_values))
                tmp_topki, tmp_topkv = utils.topk(tmp_c, k)
                first_array_indexes = tmp_topki[tmp_topki < k]
                second_array_indexes = tmp_topki[tmp_topki >= k]-k
                indexes = np.concatenate((indexes[first_array_indexes], tmp_indexes[second_array_indexes]))
                values = np.concatenate((values[first_array_indexes], tmp_values[second_array_indexes]))

                #tensor[tmp_indexes] += tmp_values
                #logger.debug('[round:%d][rank:%d] tensor: %s', i, rank, tensor)
                #indexes, values = topk(tensor, k)

                send_values = np.concatenate((indexes, values))#.astype(np.float32)
                send_values[0:k] = indexes.astype(np.uint32)
                send_values[k:] = values.astype(np.float32)
            else:
            #    logger.debug('[%d]send k: %d', rank, send_values.size)
                target = participate_ranks[local_rank-1]
                logger.debug('[round:%d], %d(%d)->%d(%d)', i, rank, local_rank, target, local_rank-1)
                comm.Send([send_values, MPI.FLOAT], dest=target)
        exist_workers /= 2
        step *= 2
        participate_ranks = range(0, num_workers, step)
        comm.Barrier()

    if rank == 0:
        #send_values = np.concatenate((indexes, values))
        #send_values[0:k] = indexes.astype(np.uint32)
        #send_values[k:] = values.astype(np.float32)
        send_indexes = indexes.astype(np.uint32)
    else:
        send_indexes = indexes.astype(np.uint32)
    comm.Bcast(send_indexes, root=0)
    allreduce_tensor = tensor[send_indexes]
    results = np.zeros_like(allreduce_tensor)
    comm.Allreduce(allreduce_tensor, results, MPI.SUM)
    tensor.fill(0.0)
    tensor[send_indexes] = results
    included_indexes = send_indexes.astype(np.int)
    return tensor, included_indexes 


def sequence_sparse_allreduce(comm, sparse_tensor, storage=None, indexes=None, dtype=np.float32):
    tensor = sparse_tensor
    start_index, k = indexes[0], indexes[1]
    send_values = tensor[start_index: start_index+k]
    if storage is not None and 'result_v3' in storage:
        recv_values = storage['result_v3']
        if recv_values.size > k:
            recv_values = recv_values[0:k] 
    else:
        recv_values = np.zeros_like(send_values)
        if storage:
            storage['result_v3'] = recv_values
    comm.Allreduce(send_values, recv_values, MPI.SUM)
    #logger.info('[rank:%d]Before norm: %f', comm.rank, np.linalg.norm(sparse_tensor))
    sparse_tensor[start_index: start_index+k] = recv_values
    #logger.info('[rank:%d]Total: %d, Start_index %d, k: %d, norm: %f', comm.rank, sparse_tensor.size, start_index, k, np.linalg.norm(sparse_tensor))
    return sparse_tensor, None

def dense_allreduce(comm, tensor):
    result = np.zeros_like(tensor)
    op = MPI.SUM
    comm.Allreduce(tensor, result, op)
    return result

def _default_err_callback(new_num_workers, new_rank):
    logger.error('Some process error accurs, number of workers changes to %d, my rank changes to %d', new_num_workers, new_rank)


class AllReducer():
    def __init__(self, named_parameters, lock, key_lock, compression, sparse=False, err_callback=None, layerwise_times=None, sigma_scale=2.5, density=0.001, train_epoch=0, norm_clip=None, msg_queue=None, msg_queue2=None, writer=None, seq_layernames=None):
        self._running = False 
        self._msg_queue = msg_queue
        self._msg_queue2 = msg_queue2
        self._writer = writer
        self._profiling = True
        self._entries = {}
        self._keys = []
        self._outputs = {}
        self._residuals = {}
        self._sparse_storages = {}
        self._sparse_storages_topk = {}
        self._sparse = sparse
        self._sigma_scale = sigma_scale
        self._density = density
        self.train_epoch = train_epoch
        self.train_iter = 0
        logger.info('density: %f', self._density)
        self._comm = MPI.COMM_WORLD
        self._comm.Set_errhandler(MPI.ERRORS_RETURN)
        self._seq_layernames = seq_layernames
        self._layerwise_times = layerwise_times # L->1: Note that the layerwise time is from the last layer to the first
        _named_parameters = list(named_parameters)
        self._named_parameters = {k: v for k, v
                                in _named_parameters}
        self._default_for_reductions = {k: 1 for k, v
                                in _named_parameters}
        self._sequential_keys = [k for k, v in _named_parameters]
        self._lock = lock
        self._key_lock = key_lock
        self._compression = compression
        self._err_callback = err_callback if err_callback else _default_err_callback
        self._norm_clip = norm_clip
        self._layerwise_compressors = None
        self._original_layerwise_times_kv = None
        if self._layerwise_times is not None and self._seq_layernames is not None:
            self._original_layerwise_times_kv = dict(zip(self._seq_layernames, self._layerwise_times))
        self._generate_merged_parameters()
        self.allocate_sparse_storages()

        self._allreduce_timers = {}
        self._compression_timers = {}
        self._merge_timers = {}
        self._demerge_timers = {}
        self._h2d_times = {}
        self._d2h_times = {}
        self._profiling_norms = {}

        #self._dynamic_densities = [0.1, 0.0725, 0.035, 0.03]
        #self._dynamic_densities = [0.25, 0.0625, 0.015625, 0.004, 0.001] # the setting used in DGC
        if self._density < 0.004:
            self._dynamic_densities = [0.1, 0.0725, 0.035, 0.02, 0.01, 0.0085, 0.0075, 0.004]
        elif self._density < 0.03:
            self._dynamic_densities = [0.1, 0.0725, 0.035]
        else:
            self._dynamic_densities = [0.1]
        self._dynamic_densities = [0.25, 0.0625, 0.015625, 0.004, 0.001] # the setting used in DGC
        #self._dynamic_densities = [0.015625, 0.004, 0.001]
        #self._dynamic_densities = [0.1, 0.08, 0.0725, 0.05, 0.035, 0.02, 0.01, 0.0085, 0.0075, 0.005]  # customized one
        #self._dynamic_densities = [0.1, 0.08, 0.0725, 0.05, 0.035]#, 0.02, 0.01]
        self._dynamic_densities = None 
        if self._dynamic_densities is not None:
            self._dynamic_densities.append(self._density)
            logger.info('dynamic densities = %s', self._dynamic_densities)
        self.reset()

        self.allreduce_count = 0

    def _allreduce_time_with_size(self, size):
        return alpha + 1e-9 * size * 4

    def _generate_groups_mgwfbp(self):
        def __calculate_comm_start(tc, tb, taob, L):
            taoc = [0] * L 
            taoc[L-1] = taob[L-1] + tb[L-1]
            for l in range(L-1)[::-1]:
                taoc[l] = max(tc[l+1] + tc[l+1], taob[l] + tb[l])
            return taoc
        def __merge(taob, tc, p, l):
            tc[l] = 0
            p[l-1] = p[l-1]+p[l]
            p[l] = 0
            tc[l-1] = self._allreduce_time_with_size(p[l-1])
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse order
        self._sizes = sizes
        p = sizes[:]
        L = len(sizes)
        tc = [self._allreduce_time_with_size(s) for s in sizes]
        tb = list(self._layerwise_times)
        taob = [0]
        for t in tb[:-1]:
            taob.append(t+taob[-1])
        taoc = __calculate_comm_start(tc, tb, taob, L)
        groups = []
        group = []
        idx = 0
        key_groupidx_maps = {}
        l = L-1
        key = self._sequential_keys[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        for l in range(1, L-1)[::-1]:
            key = self._sequential_keys[l]
            group.append(key)
            key_groupidx_maps[key] = idx
            current_taob = taob[l-2] if l >=2 else taob[0]
            if current_taob- taoc[l] < alpha:
                __merge(taob, tc, p, l)
                taoc = __calculate_comm_start(tc, tb, taob, L)
            else:
                idx += 1
                groups.append(group)
                group = []
        l = 0
        key = self._sequential_keys[l]
        key_groupidx_maps[key] = idx
        group.append(key)
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps

    def _generate_groups_with_threshold(self, threshold):
        sizes = [self._named_parameters[k].data.numel() for k in self._sequential_keys][::-1] # reverse order
        self._sizes = sizes
        logger.info('Number of params: %d', np.sum(sizes))
        sub_size = 0
        groups = []
        group = []
        key_groupidx_maps = {}
        idx = 0
        for k in self._sequential_keys[::-1]:
            numel = self._named_parameters[k].data.numel()
            sub_size += numel
            key_groupidx_maps[k] = idx
            if sub_size < threshold:
                group.append(k)
            else:
                idx += 1
                group.append(k)
                groups.append(group)
                group = []
                sub_size = 0
        if len(group) > 0:
            groups.append(group)
        return groups, key_groupidx_maps


    def _generate_merged_parameters(self):
        self._merged_parameters = {}
        if False and self._layerwise_times is not None:
            groups, key_groupidx_maps = self._generate_groups_mgwfbp()
        else:
            groups, key_groupidx_maps = self._generate_groups_with_threshold(THRESHOLD)
        logger.info('groups: %s', groups)
        logger.info('key_groupidx_maps: %s', key_groupidx_maps)
        new_keys = []
        self._merged_parameter_offsets = {}
        self._layerwise_compressors = None
        if self._original_layerwise_times_kv is not None and LAYERWISE and ADAPTIVE:
            self._layerwise_compressors = {}
        num_of_workers = self._comm.size
        for g in groups:
            sub_size = 0
            offsets = []
            computation_time = 0
            for k in g:
                offsets.append(sub_size)
                numel = self._named_parameters[k].data.numel()
                sub_size += numel
                if self._original_layerwise_times_kv is not None and k in self._original_layerwise_times_kv and LAYERWISE and ADAPTIVE:
                    computation_time += self._original_layerwise_times_kv[k]
            new_key = ':'.join(g)
            new_keys.append(new_key)
            self._merged_parameters[new_key] = torch.zeros(sub_size, device=self._named_parameters[g[0]].device, dtype=self._named_parameters[g[0]].dtype, requires_grad=False)
            self._merged_parameter_offsets[new_key] = offsets
            density = utils.predict_density_with_size_and_computation(sub_size, computation_time, num_of_workers)
            if self._layerwise_compressors is not None:
                self._layerwise_compressors[new_key] = density
        self._groups = groups
        self._key_groupidx_maps = key_groupidx_maps
        self._groups_flags = []
        for g in self._groups:
            flags = []
            for k in g:
                flags.append(0)
            self._groups_flags.append(flags)
        logger.info('offsets: ', self._merged_parameter_offsets)
        logger.info('_layerwise_compressors: %s', self._layerwise_compressors)

    def _push_to_buffer(self, name, tensor):
        with torch.no_grad():
            if len(self._groups) == len(self._sequential_keys):
                new_tensor = tensor.data.view(-1)
                return name, new_tensor 
            group_idx = self._key_groupidx_maps[name]
            g = self._groups[group_idx]
            new_key = ':'.join(g)
            layer_idx = g.index(name)
            offset = self._merged_parameter_offsets[new_key][layer_idx]
            numel = tensor.data.numel()
            self._merged_parameters[new_key].data[offset:offset+numel] = tensor.view(numel).data
            self._groups_flags[group_idx][layer_idx] = 1
            try:
                idx = self._groups_flags[group_idx].index(0)
            except:
                idx = -1
            if idx >= 0:
                return name, None
            return new_key, self._merged_parameters[new_key]

    def _pull_from_buffer(self, name, merged_tensor):
        if len(self._groups) == len(self._sequential_keys):
            shape = self._named_parameters[name].data.shape
            return {name: merged_tensor.view(shape)} 
        offsets = self._merged_parameter_offsets[name]
        g = name.split(':')
        group_idx = self._key_groupidx_maps[g[0]]
        self._groups_flags[group_idx] = [0]*len(self._groups_flags[group_idx])
        tensors = {}
        for i, k in enumerate(g):
            offset = offsets[i]
            original_tensor = self._named_parameters[k]
            numel = original_tensor.numel()
            tensor = torch.zeros(numel, device=original_tensor.device, dtype=original_tensor.dtype)
            tensor.data = merged_tensor.data[offset:offset+numel]
            tensors[k] = tensor.view(original_tensor.shape)
        return tensors

    def rank(self):
        return self._comm.rank
    
    def size(self):
        return self._comm.size

    def allocate_sparse_storages(self):
        for k, v in self._merged_parameters.items():
            self.allocate_storage(k, v)

    def _print_profiling(self):
        if self._profiling and self.rank() == 0 and len(self._allreduce_timers.keys()) > 0 and len(self._allreduce_timers.get(self._allreduce_timers.keys()[0], [])) == 100:
            cts = self._layerwise_times # gpu computation
            mgs = self._merge_timers # merge_times
            cps = self._compression_timers # compression
            ars = self._allreduce_timers # allreduce times
            dms = self._demerge_timers# demerge times
            d2hs = self._d2h_times
            h2ds = self._h2d_times
            l = 0
            logger.info('[rank:%d]name[size]: backward, merge, compression, allreduce, demerge, d2h, h2d')
            total_sz = 0
            total_ct = 0.0
            total_mg = 0.0
            total_cp = 0.0
            total_ar = 0.0
            total_dm = 0.0
            total_d2h = 0.0
            total_h2d = 0.0

            for g in self._groups:
                ct = 0.0
                sz = 0
                for k in g:
                    if cts is not None:
                        ct += cts[l]
                    else:
                        ct = 0.0
                    sz += self._sizes[l]
                    total_ct += ct
                    l += 1
                total_sz += sz
                k = ':'.join(g)
                mg = np.mean(mgs[k])
                total_mg += mg
                cp = np.mean(cps[k])
                total_cp += cp
                ar = np.mean(ars[k])
                total_ar += ar
                dm = np.mean(dms[k])
                total_dm += dm
                d2h = np.mean(d2hs.get(k, [0.0]))
                total_d2h += d2h
                h2d = np.mean(h2ds.get(k, [0.]))
                total_h2d += h2d
                key_name = k
                if len(key_name) > 64:
                    key_name = k[0:10]+'...'+k[-10:]


                logger.info('[rank:%d]%s[%d]: %f,%f,%f,%f,%f,%f,%f', self.rank(), key_name, sz, ct,mg,cp,ar,dm,d2h,h2d)
                mgs.pop(k, None)
                cps.pop(k, None)
                ars.pop(k, None)
                dms.pop(k, None)
                d2hs.pop(k, None)
                h2ds.pop(k, None)
            logger.info('[rank:%d]%s[%d]: %f,%f,%f,%f,%f,%f,%f', self.rank(), 'total', total_sz, total_ct,total_mg,total_cp,total_ar,total_dm,total_d2h,total_h2d)

    def reset(self):
        self._for_reductions = self._default_for_reductions.copy()
        self._print_profiling()

    def add_tensor(self, name, tensor):
        if name in self._entries:
            return
        self._entries[name] = tensor
        return name

    def get_current_density(self, name=None):
        density = self._density
        if self._dynamic_densities is not None:
            if self.train_epoch >= len(self._dynamic_densities):
                density = self._dynamic_densities[-1]
            else:
                density = self._dynamic_densities[self.train_epoch]
        if False and name is not None and self._layerwise_compressors is not None:
            if name not in self._layerwise_compressors:
                errstr = 'compressor density not found at layer: %s' % name
                logger.error(errstr)
                raise Exception(errstr)
            ld = self._layerwise_compressors[name]
            density = max(ld, density)
        return density


    def get_result(self, name):
        return self._outputs[name]

    def allocate_storage(self, name, tensor):
        storage = {}
        self._sparse_storages[name] = storage
        self._sparse_storages_topk[name] = {}

    def _sparse_allreduce(self, name, tensor, selected_tensor, original_shape, topk_indexes=None):
        stime = time.time()
        ct = selected_tensor
        if ct.is_cuda: # only transfer the selected k values through PCI-e
            entry = ct.data.cpu().numpy()
        else:
            entry = ct.data.numpy()
        if self._profiling:
            utils.force_insert_item(self._d2h_times, name, time.time()-stime)
        normal = True

        result = None
        included_indexes = None
        full_mean = None
        full_var = None
        #try:
        if True:
            if self._compression.name in ['topk', 'topk2', 'sigmathresallgather']:
                result, global_indexes, included_indexes = topk_sparse_allreduce(self._comm, entry, self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)
            elif self._compression.name in ['gtopk', 'centralzero', 'sigmathres']:
                result, global_indexes, included_indexes = gtopk_sparse_allreduce(self._comm, entry, storage=self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)
            elif self._compression.name == 'meangtopk':
                result, global_indexes, included_indexes, full_mean, full_var = meangtopk_sparse_allreduce(self._comm, entry, storage=self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)
            elif self._compression.name == 'gtopk2':
                result, global_indexes, included_indexes = gtopk2_sparse_allreduce(self._comm, entry, storage=self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)
            elif self._compression.name == 'atopk':
                result, included_indexes = atopk_sparse_allreduce(self._comm, entry, storage=self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)
            elif self._compression.name == 'sequence':
                result, included_indexes = sequence_sparse_allreduce(self._comm, entry, storage=self._sparse_storages[name], indexes=topk_indexes, dtype=np.float32)
        #except Exception as e:
        #    logger.error('sparse_allreduce Error: %s' % e)
        #    self._comm = self._comm.EXT_shrink()
        #    self._err_callback(self._comm.size, self._comm.rank)
        #    result = entry 
        #    normal = False

        r = torch.from_numpy(result)
        gi = torch.from_numpy(global_indexes.astype(np.int64))
        stime = time.time()
        if tensor.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
            final_indexes = gi.cuda(tensor.device, non_blocking=False)
        else:
            final_indexes = gi 

        tensor.fill_(0.0)
        if self._compression.name in ['gtopk', 'gtopk2', 'centralzero', 'sigmathres']:
            tensor[final_indexes] = r
        elif self._compression.name in ['meangtopk']:
            num_workers = self._comm.size
            #tensor.fill_(full_mean)
            random = torch.distributions.normal.Normal(full_mean, full_var)
            torch.seed(self.train_iter)
            tensor.data = random.sample(tensor.size()).data
            self.train_iter += 1
            if self._comm.rank == 0:
                logger.info('full_mean: %f', full_mean)
            tensor[final_indexes] = r
        elif self._compression.name in ['topk', 'topk2', 'sigmathresallgather']:
            num_workers = self._comm.size
            nnz = topk_indexes.size(0)
            for i in range(num_workers):
                index = final_indexes[i*nnz:(i+1)*nnz]
                tensor[index] += r[i*nnz:(i+1)*nnz]
            if self._compression.name == 'topk2':
                values, indexes = torch.topk(torch.abs(tensor.data), k=nnz)
                cv, c1, c2 = np.intersect1d(indexes.cpu().numpy(), topk_indexes.cpu().numpy(), assume_unique=False, return_indices=True)
                included_indexes = c2
                values = tensor.data[indexes]
                tensor.data.fill_(0.0)
                tensor.data[indexes] = values.data

        if normal:
            tensor /= self.size()
        if self._profiling:
            utils.force_insert_item(self._h2d_times, name, time.time()-stime)
        return tensor, included_indexes, full_mean

    def _dense_allreduce(self, name, tensor):
        stime = time.time()
        ct = tensor 
        #ct/=self.size()
        #return ct
        shape = tensor.shape
        if ct.is_cuda:
            entry = ct.data.cpu().numpy()
        else:
            entry = ct.data.numpy()
        if self._profiling:
            utils.force_insert_item(self._d2h_times, name, time.time()-stime)
        normal = True
        try:
            result = dense_allreduce(self._comm, entry)
        except Exception as e:
            logger.info('-----------dense_allreduce Error: %s' % e)
            self._comm = self._comm.EXT_shrink()
            self._err_callback(self._comm.size, self._comm.rank)
            result = entry 
            normal = False
        result = result.reshape(shape)

        stime = time.time()
        r = torch.from_numpy(result)
        if tensor.is_cuda:
            r = r.cuda(tensor.device, non_blocking=False)
        if self._profiling:
            utils.force_insert_item(self._h2d_times, name, time.time()-stime)
        if normal:
            r /= self.size()
        return r 

    def _merge_tensors(self, tensors):
        pass

    def run(self):
        self._running = True
        logger.info('Allreducer thread started ...')
        while self._running:
            #if len(self._entries) == 0:
                #time.sleep(0.001)
                #continue
            name = self._msg_queue.get()
            if name == 'STOP':
                break

            if name is not None:
                tensor = self._entries[name]

                # Push the tensor to the buffer
                stime = time.time()
                new_name, new_tensor = self._push_to_buffer(name, tensor)
                if self._profiling:
                    utils.force_insert_item(self._merge_timers, new_name, time.time()-stime)

                if new_tensor is None or new_name is None:
                    continue

                self.is_cuda = new_tensor.is_cuda

                stime = time.time()


                #logger.info('name: %s', new_name)
                density = self.get_current_density(name=new_name)
                sigma_scale = utils.get_approximate_sigma_scale(density)

                if self._norm_clip is not None:
                    norm_clip = np.sqrt(1.0/self.size()) * self._norm_clip
                    norm_type = 2.0
                    param_norm = new_tensor.norm(norm_type)
                    total_norm = param_norm.item() 
                    clip_coef = norm_clip / (total_norm + 1e-6)
                    if clip_coef < 1:
                        new_tensor.mul_(clip_coef)

                original_shape = new_tensor.shape
                if density < 1:
                    new_tensor, ctx, selected_tensor = self._compression.compress(new_tensor, new_name, sigma_scale=sigma_scale, ratio=density)

                    if self.is_cuda:
                        torch.cuda.synchronize()

                    # For comparison purpose ===>
                    #residuals = self._compression.get_residuals(new_name, new_tensor)
                    if settings.LOGGING_ASSUMPTION:
                        new_tensor_add_res = new_tensor.data.clone()
                        dense_result = self._dense_allreduce(new_name, new_tensor_add_res)
                        dense_std= float(torch.std(dense_result))
                        random_indexes = torch.randperm(dense_result.size(0))

                        k = ctx.size(0)
                        with torch.no_grad():
                            rand_k = random_indexes[:k]
                            rand_k_tensor = torch.zeros_like(dense_result)
                            rand_k_tensor.data[rand_k] = dense_result.data[rand_k]
                            randk_norm = (dense_result - rand_k_tensor).norm(p=2)
                    # For comparison purpose <=== End

                if self._profiling:
                    utils.force_insert_item(self._compression_timers, new_name, time.time()-stime)

                # Allreduce on the merged gradients 
                stime = time.time()
                if self._sparse and density < 1:
                    result, included_indexes, full_mean = self._sparse_allreduce(new_name, new_tensor, selected_tensor, original_shape, topk_indexes=ctx)
                    if included_indexes is not None:
                        if full_mean is not None:
                            self._compression.add_residuals(included_indexes, new_name, full_mean)
                        else:
                            self._compression.add_residuals(included_indexes, new_name)
                    # For comparison purpose ===>
                    if settings.LOGGING_ASSUMPTION:
                        gtopk_norm = (dense_result - result).norm(p=2)
                        xnorm  = float(dense_result.norm(p=2))
                        upbound = 1.0*(result.size(0)-k)/result.size(0) * xnorm 
                        utils.force_insert_item(self._profiling_norms, new_name,(float(gtopk_norm), float(randk_norm), upbound, xnorm, dense_std))
                    # For comparison purpose <=== End

                else:
                    result = self._dense_allreduce(new_name, new_tensor)

                if self._profiling:
                    utils.force_insert_item(self._allreduce_timers, new_name, time.time()-stime)

                # Decouple on the merged gradients 
                stime = time.time()
                tensors = self._pull_from_buffer(new_name, result)
                if self._profiling:
                    utils.force_insert_item(self._demerge_timers, new_name, time.time()-stime)
                for n in tensors:
                    self._outputs[n] = tensors[n] 
                    self._entries.pop(n, None)
                    self._for_reductions.pop(n, None)

            if len(self._for_reductions) == 0:
                self.reset()
                torch.cuda.synchronize()
                self._msg_queue2.put('DONE')
           
    def stop(self):
        self._running = False


def benchmark_gtopk_sparse_allreduce():
    logger.setLevel(logging.INFO)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    #np.random.seed(rank)
    size = 25 * 1024 * 1024
    ratio = 0.001
    tensor = np.random.rand(size).astype(np.float32)
    k = int(tensor.size * ratio)
    indexes, values = utils.topk(tensor, k)
    #indexes, values = topk(tensor, k)
    #logger.info('topk[%d]%s', rank, values)
    tmp = tensor[indexes]
    tensor.fill(0.)
    tensor[indexes] = tmp
    logger.debug('[%d]%s', rank, tensor)
    storage = {}

    t = gtopk_sparse_allreduce(comm, tensor, storage=storage, indexes=indexes)
    iteration = 10
    stime = time.time()
    for i in range(iteration):
        t,_ = gtopk_sparse_allreduce(comm, tensor, storage=storage, indexes=indexes)
    total_time = time.time() - stime
    logger.info('average time: %f', total_time/iteration)

def benchmark_gtopk2_sparse_allreduce():
    logger.setLevel(logging.INFO)
    comm = MPI.COMM_WORLD
    rank = comm.rank
    #np.random.seed(rank)
    size = 40
    ratio = 0.1
    tensor = np.random.rand(size).astype(np.float32)
    k = int(tensor.size * ratio)
    indexes, values = utils.topk(tensor, k)
    tmp = tensor[indexes]
    tensor.fill(0.)
    tensor[indexes] = tmp
    entry = tmp
    logger.info('[%d]indexes: %s', rank, indexes)
    logger.info('[%d]%s', rank, tmp)
    storage = {}

    result, global_indexes, included_indexes = gtopk2_sparse_allreduce(comm, entry, storage=storage, indexes=indexes, dtype=np.float32)
    logger.info('[%d]included_indexes: %s', rank, included_indexes)
    logger.info('[%d]result: %s', rank, result)


if __name__ == '__main__':
    #benchmark_gtopk_sparse_allreduce()
    benchmark_gtopk2_sparse_allreduce()


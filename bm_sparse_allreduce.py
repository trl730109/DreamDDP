# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import time
import torch
import threading
import allreducer as ar
import compression as cp

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
gpu_count = torch.cuda.device_count()
num_workers = comm.size


compressor = 'topk'

named_parameters = {}
parameters = {}
#sizes = [2**i for i in range(3, 24)]
sizes = [2**23]
for s in sizes:
    name = 'tensor.%d' % s
    tensor = torch.rand(s, dtype=torch.float).cuda(device='cuda:%d'%(comm.rank%gpu_count))
    key = (name, tensor)
    named_parameters[key] = tensor 
    parameters[name] = tensor

allreducer = ar.AllReducer(named_parameters, threading.Lock(), threading.Lock(), cp.compressors[compressor], sparse=True)

def benchmark():
    iter = 200
    # warmup
    allreducer._sparse_allreduce(name, named_parameters[key])
    if comm.rank == 0:
        print('Size\ttime\tbandwidth')
    for s in sizes:
        st = time.time()
        for i in range(iter):
            allreducer._sparse_allreduce(name, parameters[name])
        et = time.time()
        avg_time = (et-st)/iter
        bytes = s * 4.0 / (1.e9)
        if comm.rank == 0:
            print('%d\t%.4f\t%.8f GB/s' %(s, avg_time, bytes/avg_time))
        #print('Num of workers: %d, Time used: %f, size: %d GB, bandwidth: %f GB/s' % (comm.size, avg_time, bytes, bytes / avg_time))


if __name__ == '__main__':
    benchmark()

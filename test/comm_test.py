from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import broadcast_async_
from horovod.torch.mpi_ops import synchronize

import os
import torch
import hv_distributed_optimizer as hvd
os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'
os.environ['HOROVOD_CACHE_CAPACITY'] = '0'

from profiling import CommunicationProfiler
ngpus_per_node = 4

hvd.init()
rank = hvd.rank()
torch.cuda.set_device(rank%ngpus_per_node)

comm_profiler = CommunicationProfiler(allreduce_async_, synchronize)
sizes, times = comm_profiler.benchmark(num_iters=10)

for s, t in zip(sizes, times):
    print('%d %f' % (s, t))

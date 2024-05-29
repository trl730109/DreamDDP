
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from horovod.torch.mpi_ops import allreduce_async_
from horovod.torch.mpi_ops import allgather_async
from horovod.torch.mpi_ops import broadcast_async_
from horovod.torch.mpi_ops import synchronize
from horovod.torch.mpi_ops import size, local_size, rank, local_rank
from horovod.torch.mpi_ops import init, broadcast

import torch.nn as nn
import horovod.torch as hvd
from collections import defaultdict
import time
import torch
import numpy as np
import utils
from utils import *

import collections
import settings
from settings import logger, ADAPTIVE_MERGE, ADAPTIVE_SPARSE, DEBUG
from draw_plot import *

from profiling import CommunicationProfiler
from sklearn.linear_model import LinearRegression
from compression import *
from hv_distributed_optimizer import *

# Assuming the necessary functions and classes (`allreduce_model_weights`, `MockCompressor`, etc.) are defined elsewhere

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.param = nn.Parameter(torch.randn(10)).reshape(2,5)  # Initialize parameters


def main():
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # Initialize model and compressor
    model = SimpleModel().cuda()
    compressor = TopKCompressor()
    
    # Each rank modifies its parameter to be range(10) * hvd.rank()
    values = torch.arange(10, dtype=torch.float32) * hvd.rank()
    model.param.data.copy_((values.reshape(2,5)).cuda())

    # Run the allreduce model weights function
    allreduce_model_weights(model, compressor, density=0.1, strategy='average', overlap_scalar=1.0)
    
    expected_values = torch.zeros(10).cuda()
    expected_values[-1] = torch.arange(10, dtype=torch.float32)[-1] * sum(range(hvd.size())) / hvd.size()
    actual_values = model.param.data
    
    if torch.allclose(actual_values, expected_values, atol=1e-5):
        print(f"Rank {hvd.rank()}: Test Passed. Compressed averaged values are correct.")
    else:
        print(f"Rank {hvd.rank()}: Test Failed. Expected {expected_values}, got {actual_values}")

if __name__ == "__main__":
    main()

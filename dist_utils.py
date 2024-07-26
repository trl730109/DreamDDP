import time
import torch
import numpy as np
import utils
from utils import *

import collections
import settings
import torch.distributed as dist
from settings import logger, ADAPTIVE_MERGE, ADAPTIVE_SPARSE, DEBUG
from draw_plot import *

from profiling import CommunicationProfiler
# from sklearn.linear_model import LinearRegression

def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        #handle = dist.broadcast(p, root_rank, name)
        handle = dist.broadcast(p, root_rank, async_op=True)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        handle.wait()
        
def allreduce_model_weights(model, compressor, density, strategy, overlap_scalar):
    if isinstance(model, dict):
        state_dict = model
    else:
        state_dict = model.state_dict()

    params = []
    callbacks = {}
    occurrences = collections.defaultdict(int)

    # Returns the full type structure of the possibly nested objects for recursive casting back
    def _get_types(x):
        if isinstance(x, collections.Iterable):
            return type(x), [_get_types(xi) for xi in x]
        else:
            return type(x)

    # Casts an object encoded in a tensor back into its original type and subtypes
    def _recursive_cast(x, dtype):
        if isinstance(dtype, tuple):
            t, dtypes = dtype
            x = t(x)
            return t([_recursive_cast(x[i], dtypes[i]) for i in range(len(x))])
        else:
            return dtype(x)

    def _create_callback(name, t, p):
        def _from_tensor():
            state_dict[name] = t(p.numpy()[0])
        return _from_tensor


    for name, p in state_dict.items():
 
        occurrences[name] += 1
        key = '%s.%d' % (str(name), occurrences[name])
        if not torch.is_tensor(p):
            t = type(p)
            p = torch.Tensor([p])
            callbacks[key] = _create_callback(name, t, p) #create the function that transfers the scalar back to the original type

        params.append((key, p))

    if (density < 1):
        allgather_parameters_compressed(params, compressor, density, strategy, overlap_scalar)
    else:
        allgather_parameters(params,strategy)
  
    for key, p in params:
        #logger.info(f"p is {p}")
        if key in callbacks:
            callbacks[key]()
    return params

def allgather_parameters_compressed(params, compressor, density, strategy, overlap_scalar):
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # Support both named_parameters() and regular parameters
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError(f'Invalid params type: {type(params)}')

    # Run asynchronous gathers
    handles = []
    compressor.clear()
    for name, param in params:
        flat_tensor = compressor.flatten(param.data, name=name)
        _, indexes, compressed_param = compressor.compress(flat_tensor, name=name, ratio=density)

        gathered_tensors = torch.empty([dist.get_world_size(), compressed_param.numel()], dtype=compressed_param.dtype, device=compressed_param.device)
        gathered_indexes = torch.empty([dist.get_world_size(), indexes.numel()], dtype=indexes.dtype, device=indexes.device)

        handle = dist.all_gather(gathered_tensors, compressed_param, async_op=True)
        handle_idx = dist.all_gather(gathered_indexes, indexes, async_op=True)
        handles.append((handle, handle_idx, param, gathered_tensors, gathered_indexes, name))

    # Wait for completion and decompress
    for handle, handle_idx, param, gathered_tensors, gathered_indexes, name in handles:
        handle.wait()
        handle_idx.wait()
        
        decompressed_params = []
        for i in range(dist.get_world_size()):
            start_index = i * compressed_param.numel()
            end_index = start_index + compressed_param.numel()
            decompressed = compressor.decompress(
                gathered_tensors[start_index:end_index].view_as(compressed_param),
                gathered_indexes[start_index:end_index].view_as(indexes),
                shape=param.shape
            )
            decompressed_params.append(decompressed)

        averaged_param = sum(decompressed_params) / dist.get_world_size()
        param.data.copy_(averaged_param)

def allgather_parameters(params, strategy):
    if isinstance(params, dict):
        params = list(params.items())
    elif isinstance(params, list):
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError(f'Invalid params type: {type(params)}')

    handles = []
    # Start asynchronous gathering of parameters
    for name, param in params:
        tensor_list = [torch.zeros(param.view(-1).shape, dtype=param.dtype, device=param.device) for _ in range(dist.get_world_size())]
        handle = dist.all_gather(tensor_list, param.data.view(-1), async_op=True)
        handles.append((handle, param, tensor_list, name))

    for handle, param, tensor_list, name in handles:
        handle.wait()
        #gathered_results[name] = (param, tensor_list)
        new_params = torch.zeros_like(param, dtype=torch.float32, device=param.device).view(-1)
        for tensor in tensor_list:
            new_params += tensor
        new_params /= dist.get_world_size()
        param.data.copy_(new_params.view(param.shape))



def allgather_layers(model, strategy, layer_name_list):
    state_dict = model
    handles = []
    for name in layer_name_list:
        p = state_dict[name]
        tensor_list = [torch.zeros(p.view(-1).shape, dtype=p.dtype, device=p.device) for _ in range(dist.get_world_size())]
        handle = dist.all_gather(tensor_list, p.data.view(-1), async_op=True)
        handles.append((handle, p, tensor_list, name))

    for handle, p, tensor_list, name in handles:
        handle.wait()
        #gathered_results[name] = (param, tensor_list)
        new_params = torch.zeros_like(p, dtype=torch.float32, device=p.device).view(-1)
        for tensor in tensor_list:
            new_params += tensor
        new_params /= dist.get_world_size()
        #p.data.copy_(new_params.view(p.shape))

        state_dict[name] = new_params.view(p.shape)
    else:
        pass
    
    return state_dict

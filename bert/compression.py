# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import numpy as np
import time
import math
# from dear import utils
import utils
from scipy import stats
from comm_core import rank, size, Communicator, init as comm_init

# MSTopk
import tcmm
import batched_tcmm

class NoneCompressor():
    def __init__(self):
        self.name = 'none'

    def compress(self, tensor):
        return tensor, tensor.dtype

    def decompress(self, tensor, ctc):
        z = tensor 
        return z 


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    def __init__(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 
        self.c = 0
        self.t = 0.
        self.name = 'topk'
        self.zc = None
        self.current_ratio = 1

    def _process_data_before_selecting(self, name, data):
        pass

    # 处理残差数据之后，生成一个全为1的张量，并在指定的索引位置置为0，然后将处理后的张量赋值给属性self.zc。
    def _process_data_after_residual(self, name, data):
        if name not in self.zero_conditions:
            self.zero_conditions[name] = torch.ones(data.numel(), dtype=torch.float32, device=data.device)
        zero_condition = self.zero_conditions[name]
        zero_condition.fill_(1.0)
        zero_condition[self.indexes[name]] = 0.0
        self.zc = zero_condition



    def clear(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 

    # def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):  #ratio=0.05
    #     start = time.time()
    #     with torch.no_grad():
    #         if name not in self.residuals:
    #             self.residuals[name] = torch.zeros_like(tensor.data)
    #         # print('self.residuals=\n', self.residuals)
    #         # top-k solution
    #         numel = tensor.numel()
    #         k = max(int(numel * ratio), 1)
    #         # print('k=\n', k)
    #         self.current_ratio = ratio
    #         self._process_data_before_selecting(name, tensor.data)
    #
    #         values, indexes = torch.topk(torch.abs(tensor.data), k=k)
    #         values = tensor.data[indexes]
    #
    #         self.residuals[name].data = tensor.data + 0.0
    #         self.residuals[name].data[indexes] = 0.
    #         self.values[name] = values
    #         self.indexes[name] = indexes
    #
    #         self._process_data_after_residual(name, tensor.data)
    #
    #       #   # #pj添加
    #       #   tensor.data.add_(self.residuals[name].data)  #累加residuals，修复之前的压缩结果
    #       #   values, indexes = torch.topk(torch.abs(tensor.data), k=k)
    #       #   values = tensor.data[indexes] #???
    #       #
    #       #
    #       #   if name not in self.zero_conditions: #创建一个全1张量，用于表示元素是否为0的条件
    #       #       self.zero_conditions[name] = torch.ones(numel, dtype=torch.float32, device=tensor.device)
    #       #   zero_condition = self.zero_conditions[name] #获取之前创建的条件张量
    #       #   zero_condition.fill_(1.0) #将条件张量中的所有元素填充为1
    #       #   zero_condition[indexes] = 0.0 #将对应索引处的元素值置为0，表示这些元素被压缩为零
    #       #
    #       #   self.residuals[name].data.fill_(0.)  # 将之前存储压缩结果的张量清零
    #       #   self.residuals[name].data = tensor.data * zero_condition  # 修复后的压缩结果
    #       #   tensor.data.sub_(self.residuals[name].data)  # 将修复后的压缩结果从输入张量中减去，得到最终的压缩结果
    #       #
    #       #   # self.residuals[name].data = tensor.data + 0.0  #tensor.data转为浮点数
    #       #   # self.residuals[name].data[indexes] = 0. #清除residual中indexes地方的值 清零
    #       #
    #       #   # #pj添加
    #       #   # tensor.data.sub_(self.residuals[name].data)
    #       #   # # pj
    #       #
    #       #   self.values[name] = values
    #       #   self.indexes[name] = indexes
    #       #
    #       # #PJ注释掉源代码
    #       #   # self._process_data_after_residual(name, tensor.data)
    #       #    # #pj添加
    #
    #         return tensor, indexes, values #返回压缩后的张量tensor，以及被置为0的元素的索引indexes和对应的值values。



    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05, i=None, pad_grad = None):  #ratio=0.05
        start = time.time()
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(pad_grad.data)
            # top-k solution
            n = tensor.numel()
            k = max(int(n * ratio), 1)
            # self.current_ratio = ratio
            # self._process_data_before_selecting(name, tensor.data)
            if rank()==0:
                if tensor.data.shape!=self.residuals[name].data[i*n:(i+1)*n].shape:
                    print("shape not equal")
                    print(name)
            tensor.data.add_(self.residuals[name].data[i * n:(i + 1) * n])  # 累加residuals，修复之前的压缩结果

            # values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values, indexes = tcmm.f_topk(torch.abs(tensor.data), k) 
            # values, indexes = batched_tcmm.f_batched_topk(torch.abs(tensor.data), k) 
            indexes = indexes.to(dtype=torch.int64)
            values = tensor.data[indexes]

            start = i*n
            end = start + n
            self.residuals[name].data[start:end] = tensor.data + 0.0
            self.residuals[name].data[start:end][indexes] = 0.
            tensor.data.sub_(self.residuals[name].data[start:end])  # 将修复后的压缩结果从输入张量中减去，得到最终的压缩结果
            # self.values[name] = values
            # self.indexes[name] = indexes

            # self._process_data_after_residual(name, tensor.data)
            return tensor, indexes, values #返回压缩后的张量tensor，以及被置为0的元素的索引indexes和对应的值values。


    def get_residuals(self, name, like_tensor):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(like_tensor.data)
        return self.residuals[name]

    def add_residuals(self, included_indexes, name):
        with torch.no_grad():
            residuals = self.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            else:
                indexes_t = included_indexes
            values = self.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[self.indexes[name]] += values.data

    def decompress(self, tensor, original_tensor_size):
        return tensor


class EFTopKCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'eftopk'

    def _process_data_before_selecting(self, name, data):
        data.add_(self.residuals[name].data)


#import bit2byte
class SignCompressor:
    """Taken from https://github.com/PermiJW/signSGD-with-Majority-Vote"""
    def __init__(self):
        self.zc = None
        self.name = 'signum'

    def _process_data_before_selecting(self, name, data):
        pass

    def _process_data_after_residual(self, name, data, original_tensor):
        pass

    def packing(self, src_tensor):
        src_tensor = torch.sign(src_tensor)
        packed_data = src_tensor
        src_tensor_size = src_tensor.size()
        src_tensor = src_tensor.view(-1)
        src_len = len(src_tensor)
        add_elm = 32 - (src_len % 32)
        if src_len % 32 == 0:
            add_elm = 0
        new_tensor = torch.zeros([add_elm], dtype=torch.float32, device=src_tensor.device)
        src_tensor = torch.cat((src_tensor, new_tensor), 0)
        src_tensor = src_tensor.view(32, -1)
        src_tensor = src_tensor.to(dtype=torch.int32)
        dst_tensor = bit2byte.packing(src_tensor)
        dst_tensor = dst_tensor.to(dtype=torch.int32)
        return dst_tensor, packed_data

    def unpacking(self, src_tensor, src_tensor_size):
        src_element_num = self.element_num(src_tensor_size)
        add_elm = 32 - (src_element_num % 32)
        if src_element_num % 32 == 0:
            add_elm = 0
        src_tensor = src_tensor.int()
        new_tensor = torch.ones(
            src_element_num + add_elm, device=src_tensor.device, dtype=torch.int32
        )
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = new_tensor.view(-1)
        new_tensor = new_tensor[:src_element_num]
        new_tensor = new_tensor.view(src_tensor_size)
        new_tensor = -new_tensor.add_(-1)
        new_tensor = new_tensor.float()
        return new_tensor

    def majority_vote(self, src_tensor_list):
        voter_num = len(src_tensor_list)
        src_tensor = torch.stack(src_tensor_list)
        src_tensor = src_tensor.view(-1)
        full_size = 32 * len(src_tensor)
        new_tensor = torch.ones(full_size, device=src_tensor.device, dtype=torch.int32)
        new_tensor = new_tensor.view(32, -1)
        new_tensor = bit2byte.unpacking(src_tensor, new_tensor)
        new_tensor = -new_tensor.add_(-1)
        # sum
        new_tensor = new_tensor.permute(1, 0).contiguous().view(voter_num, -1)
        new_tensor = torch.sum(new_tensor, 0)
        new_tensor = new_tensor.view(-1, 32).permute(1, 0)
        new_tensor = torch.sign(new_tensor)
        new_tensor = bit2byte.packing(new_tensor)
        new_tensor = new_tensor.to(dtype=torch.int32)
        return new_tensor

    def element_num(self, size):
        num = 1
        for i in range(len(size)):
            num *= size[i]
        return num

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        self._process_data_before_selecting(name, tensor)
        packed_tensor, packed_data = self.packing(tensor)
        self._process_data_after_residual(name, packed_data, tensor)
        return packed_tensor, None, None

    def decompress(self, tensor, original_tensor_size):
        dst_tensor = self.unpacking(tensor, original_tensor_size)
        return dst_tensor


class EFSignCompressor(SignCompressor):
    def __init__(self):
        super().__init__()
        self.zc = None
        self.name = 'efsignum'
        self.residuals = {}

    def _process_data_before_selecting(self, name, data):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(data)
        data.add_(self.residuals[name].data)

    def _process_data_after_residual(self, name, packed_data, original_tensor):
        self.residuals[name] = original_tensor - packed_data


class GaussianCompressor(TopKCompressor):
    """
    """

    def __init__(self):
        super().__init__()
        self.name = 'gaussian'
        self.iterations = {}
        self.sparsities = []

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            tensor.add_(self.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 3:
                one_indexes = abs_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2*k/3:
                    right_thres *= 0.5
                elif indexes.numel() > 4*k/3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            indexes = indexes[0:k]
            values = tensor.data[indexes] 
            #print('gaussion vs topk: ', indexes.numel(), k)
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values


compressors = {
        'none': NoneCompressor,
        None: NoneCompressor,
        'topk': TopKCompressor,
        'eftopk': EFTopKCompressor, #TopK with error-feedback
        'gaussian': GaussianCompressor, #GaussianK with error-feedback

        'signum': SignCompressor,
        'efsignum': EFSignCompressor,
        }

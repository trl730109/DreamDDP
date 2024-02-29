# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
import numpy as np

def test():
    size = 60*1024*1024
    MB = 1024*1024
    tensor = torch.rand(size).float().cuda()
    torch.cuda.synchronize()
    niter = 20
    stime = time.time()
    for i in range(20):
        t = tensor.cpu().numpy()
    used_time = (time.time() - stime)/niter
    print('size: %d MB, time: %f, bandwidth: %f MB/s' % (size*4/MB, used_time, size*4/MB/used_time))


    for s in np.arange(1024, size, step=4096):
        nt = tensor[0:s]
        stime = time.time()
        for i in range(20):
            t = nt.cpu().numpy()
        used_time = (time.time() - stime)/niter
        print('size: %f MB, time: %f, bandwidth: %f MB/s' % (nt.numel()*4./MB, used_time, nt.numel()*4./MB/used_time))



if __name__ == '__main__':
    test()

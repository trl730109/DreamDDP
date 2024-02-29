# -*- coding: utf-8 -*-
from __future__ import print_function
import timeit
import random
from datasets import DatasetHDF5
from six.moves import reduce

MODE='random'

#hdf5fn = '/home/datasets/imagenet/imagenet_hdf5/imagenet-shuffled.hdf5'
hdf5fn = '/tmp/imagenet_hdf5/imagenet-shuffled.hdf5'
dataset = DatasetHDF5(hdf5fn, 'train')
nsamples = len(dataset)
img_list = list(range(nsamples))
batch_size = 16
shape = dataset[0][0].shape
size = reduce(lambda x, y: x*y, list(shape))
size_batch = batch_size * size * 4
MB = 1024*1024
print('size: ', size_batch)


def sequence_read():
    index = random.randint(0, nsamples-batch_size)
    current_list = img_list[index: index+batch_size]
    for i in range(batch_size):
        idx = current_list[i]
        d = dataset[idx]

    
def random_read():
    random.shuffle(img_list)
    index = random.randint(0, nsamples-batch_size)
    current_list = img_list[index: index+batch_size]
    for i in range(batch_size):
        idx = current_list[i]
        d = dataset[idx]
        #print('d.shape: ', d.shape)

func = sequence_read
if MODE == 'random':
    func = random_read
# warmup 
timeit.timeit(func, number=20)

num_iters = 2000
elapsed_time = timeit.timeit(func, number=num_iters)
batch_io_time = elapsed_time/num_iters
print('Elapsed time: ', batch_io_time)
print('Bandwidth: %f MB/s' % (size_batch/batch_io_time/MB))



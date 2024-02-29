# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import scipy.special as spe 
import utils as u
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

def x_topkx_diff(sigma, thres):
    const = 1.0 / np.sqrt(2*np.pi* sigma**2)
    right = -sigma**2 * thres * np.exp(-thres**2/(2*sigma**2)) + sigma**4 * np.sqrt(np.pi) * spe.erf(thres/(2*sigma**2))
    left = sigma**2 * thres * np.exp(-thres**2/(2*sigma**2)) + sigma**4 * np.sqrt(np.pi) * spe.erf(-thres/(2*sigma**2))
    val = const * (right - left)
    return val


def topk(tensor, k):
    indexes = np.abs(tensor).argsort()[-k:]
    return indexes, tensor[indexes]


def test_gaussian():
    d = 50000 # dimension
    density = 0.001
    k = int(density * d)
    sigma = 0.2 # sigma
    gradients = sigma * np.random.randn(d)
    topkg = np.copy(gradients)
    zero_indexes = np.abs(topkg).argsort()[0:d-k]
    topkg[zero_indexes] = 0.0

    #topkg[indexes]
    #u.plot_hist(gradients, title='gaussian', ax=ax)
    #plt.show()
    diff = x_topkx_diff(sigma, 3*sigma)
    xl2norm_sq = np.linalg.norm(gradients) ** 2
    topkl2norm_sq = np.linalg.norm(topkg) ** 2
    diff_norm_sq = np.linalg.norm(gradients - topkg) ** 2
    print('diff: ', diff)
    print('E[||x||^2]=: ', d*sigma**2)
    print('||x||^2= ', xl2norm_sq)
    print('(1-k/d)||x||^2= ', xl2norm_sq*(1.0-k*1.0/d))
    print('||topk(x)||^2= ', topkl2norm_sq)
    print('||x-topk(x)||^2= ', diff_norm_sq)


if __name__ == '__main__':
    test_gaussian()

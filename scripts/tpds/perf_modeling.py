# -*- coding: utf-8 -*-
from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import readers
import numpy as np
from scipy.optimize import curve_fit
from plot_sth import Bar
import utils as u
import plot_sth

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

MGD_INPUT_PATH = './logs'
OUTPUTPATH='/media/sf_Shared_Data/tmp/tpds'
FONTSIZE=16

s=2.18896957e-10
multi_p_ab = {
        2: (1.6e-3, 1.0e-8),
        4: (2.7e-3, 1.3e-8),
        8: (4.0e-3, 1.5e-8),
        16: (1.1e-2, 1.7e-8)
        }

def topk_perf_model(x, s=s):
    """
    x is the number of parameters
    Return: s * x * log2(x)
    """
    #if x == 0.0:
    #    return 0.0
    return s * x * np.log2(x)

def allgather_perf_model(x, P):
    """
    x is the number of parameters
    Return: t = a + b * x
    """
    a, b = multi_p_ab[P]
    if x == 0.0:
        return 0.0
    return a + b * x * 4 

def fit_topk_model(x, y):
    init_vals = [1]
    s, covar = curve_fit(topk_perf_model, x, y, p0=init_vals)
    print('s: ', s)
    return s

def topk_measured():
    #logfile = '%s/%s' % (MGD_INPUT_PATH, 'topk-p102100-nosorted.log')
    logfile = '%s/%s' % (MGD_INPUT_PATH, 'topk-v100-nosorted.log')
    sizes, times = readers.read_topkperf_log(logfile)
    fig, ax = plt.subplots()
    ax.scatter(sizes, times, label='Measured', marker='+')

    #logfile = '%s/%s' % (MGD_INPUT_PATH, 'topk-p102100.log')
    #sizes2, times2 = readers.read_topkperf_log(logfile)
    #ax.scatter(sizes2, times2, label='Measured Top-k performance')

    s = fit_topk_model(sizes, times)[0]
    ax.plot(sizes, topk_perf_model(sizes, s), label=r'Predicted ($\gamma$=%e)'%s, color='coral')
    ax.legend(fontsize=FONTSIZE)
    ax.set_xlabel('# of parameters')
    ax.set_ylabel('Top-k selection time [s]')
    u.update_fontsize(ax, FONTSIZE)
    #plt.savefig('%s/topkperf.pdf'%OUTPUTPATH, bbox_inches='tight')

def plot_merged_topk():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(1024, 1024*100, step=1024)
    y = np.arange(1024, 1024*100, step=1024)
    X, Y = np.meshgrid(x, y)
    unmerged = topk_perf_model(X) + topk_perf_model(Y)
    merged = topk_perf_model(X+Y)
    surf = ax.plot_surface(X, Y, merged, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    surf2 = ax.plot_surface(X, Y, unmerged, rstride=1, cstride=1, cmap=cm.get_cmap(),
                       linewidth=0, antialiased=False)
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)


def plot_pipeline_bars(ax, taob, tb, taos, ts, taoc, tc, max_time, start=0.3, title='Group1'):
    L = len(taob)
    indexes = []
    for l in range(0,L)[::-1]:
        start_time = taob[l]
        comp = tb[l]
        indexes.append(str(l+1))
        if comp == 0:
            continue
        index = ','.join(indexes)
        bar_comp = Bar(start_time, comp, max_time, ax, type='p', index=index, start=start, title=title)
        bar_comp.render()

        sparse_start_time = taos[l]
        sparse = ts[l]

        bar_sparse = Bar(sparse_start_time, sparse, max_time, ax, type='ps', index=index, start=start, title=title)
        bar_sparse.render()

        comm_start_time = taoc[l]
        comm = tc[l]
        bar_m = Bar(comm_start_time, comm, max_time, ax, type='wc', index=index, start=start, title=title)
        bar_m.render()
        indexes = []

def mgs_simulation():
    """
    layerwise_compute_times,
    layerwise_sizes
    """
    def __calculate_sparse_and_backward_start(tb, sizes, L, start=0):
        taos = [start] * L 
        ts = [topk_perf_model(s) for s in sizes]
        taob = [start] * L 
        taob[L-1] = start 
        taos[L-1] = taob[L-1] + tb[L-1]
        for l in range(L-1)[::-1]:
            taob[l] = taos[l+1] + ts[l+1]
            taos[l] = taob[l] + tb[l]
        return taob, taos, ts

    def __calculate_comm_start(ts, taos, sizes, L, P):
        taoc = [0] * L 
        tc = [allgather_perf_model(s, P) for s in sizes]
        taoc[L-1] = taos[L-1] + ts[L-1]
        for l in range(L-1)[::-1]:
            taoc[l] = max(taoc[l+1] + tc[l+1], taos[l] + ts[l])
        return taoc, tc


    #layerwise_compute_times = [0.01, 0.01, 0.01, 0.01]
    #layerwise_sizes = [0.2e6, 0.1e6, 0.4e6, 0.2e6]
    #P=4;dnn='vgg16';bs=128;thres=0;lr=0.1
    P=4;dnn='inceptionv4';bs=32;thres=0;lr=0.01
    #fn='/home/shshi/work/p2p-dl/logs/allreduce-comp-topk-gwarmup-dc1-model-infocom20-thestest-adamerge-thres-8kbytes/resnet50-n%d-bs16-lr0.0100-ns1-ds0.001/MGD-0.log' % (P)
    fn='/home/shshi/work/p2p-dl/logs/allreduce-comp-topk-gwarmup-dc1-model-infocom20-production-thres-%dkbytes/%s-n%d-bs%s-lr%.4f-ns1-ds0.001/MGD-0.log' % (thres, dnn, P, bs, lr)
    layerwise_sizes, layerwise_compute_times = readers.read_layerwise_times(fn)
    print('layerwise_compute_times: ', layerwise_compute_times)
    print('tb: ', np.sum(layerwise_compute_times))
    tb = list(layerwise_compute_times)
    p = list(layerwise_sizes)
    L = len(layerwise_compute_times)
    def _algorithm_layerwise(tb, p, L, P):
        taob, taos, ts = __calculate_sparse_and_backward_start(tb, p, L)
        taoc, tc = __calculate_comm_start(ts, taos, p, L, P)
        return taob, tb, taos, ts, taoc, tc

    def _algorithm_singlelayer(tb, p, L, P):
        alltb = np.sum(tb)
        tb = [0] * L 
        tb[0] = alltb
        allsize = np.sum(p)
        newp = [0] * L
        newp[0] = allsize
        taob, taos, ts = __calculate_sparse_and_backward_start(tb, newp, L)
        taoc, tc = __calculate_comm_start(ts, taos, newp, L, P)
        return taob, tb, taos, ts, taoc, tc

    def _algorithm_mgs(tb, p, L, P):
        def __merge(tb, ts, tc, p, l):
            tb[l-1] += tb[l]
            tb[l] = 0

            p[l-1] = p[l-1]+p[l]
            p[l] = 0

            tc[l-1] = allgather_perf_model(p[l-1], P) 
            tc[l] = 0

            ts[l-1] = topk_perf_model(p[l-1])
            ts[l] = 0

        L = len(tb)
        taob, taos, ts = __calculate_sparse_and_backward_start(tb, p, L)
        taoc, tc = __calculate_comm_start(ts, taos, p, L, P)
        for l in range(1, L-1)[::-1]:
            tw = tb[l-1]+topk_perf_model(p[l]+p[l-1])\
                - topk_perf_model(p[l]) - topk_perf_model(p[l-1])\
                - (taoc[l] - (taos[l]+ts[l]))
            #print('[%d], tw: %f, a: %f' % (l, tw, a))
            a = multi_p_ab[P][0]
            if tw < a:
                #print('merged: %d' % l)
                __merge(tb, ts, tc, p, l)
                taob2, taos2, ts2 = __calculate_sparse_and_backward_start(tb[:l], p[:l], l, start=taob[l]+tb[l])
                taob[:l] = taob2
                taos[:l] = taos2
                taoc, tc = __calculate_comm_start(ts, taos, p, L, P)
        return taob, tb, taos, ts, taoc, tc

    fig, ax = plt.subplots(figsize=(5*3,4.5))
    lw_taob, lw_tb, lw_taos, lw_ts, lw_taoc, lw_tc = _algorithm_layerwise(tb[:], p, L, P)
    sl_taob, sl_tb, sl_taos, sl_ts, sl_taoc, sl_tc = _algorithm_singlelayer(tb[:], p, L, P)
    print('signlelayer: ', sl_taob, sl_tb, sl_taos, sl_ts, sl_taoc, sl_tc)
    mgs_taob, mgs_tb, mgs_taos, mgs_ts, mgs_taoc, mgs_tc = _algorithm_mgs(tb[:], p, L, P)
    #print(mgs_taob, mgs_tb, mgs_taos, mgs_ts, mgs_taoc, mgs_tc)
    max_time = lw_taoc[0] + lw_tc[0] 
    max_time2 = sl_taoc[0] + sl_tc[0] 
    max_time3 = mgs_taoc[0] + mgs_tc[0] 
    max_time = max(max_time, max_time2)
    max_time = max(max_time, max_time3)

    plot_pipeline_bars(ax, lw_taob, lw_tb, lw_taos, lw_ts, lw_taoc, lw_tc, max_time, start=0.5, title='Layerwise')

    plot_pipeline_bars(ax, sl_taob, sl_tb, sl_taos, sl_ts, sl_taoc, sl_tc, max_time, start=0.3, title='SingleLayer')

    plot_pipeline_bars(ax, mgs_taob, mgs_tb, mgs_taos, mgs_ts, mgs_taoc, mgs_tc, max_time, start=0.1, title='MGS')
    
def mgwfbp_simulation():
    P=16;dnn='inceptionv4';bs=64;thres=0;lr=0.01
    fn='./logs/allreduce-gwarmup-dc1-model-tpds-prodcorrect-ada-thres-%dkbytes/%s-n%d-bs%s-lr%.4f-ns1-ds1.0/hsw224-0.log' % (thres, dnn, P, bs, lr)
    layerwise_sizes, layerwise_compute_times = readers.read_layerwise_times(fn)
    #alpha, beta = 0.00010632079996292579, 3.2713239529771973e-10
    alpha, beta = 1.5*0.000357938692551777, 1.5*3.2713239529771973e-10

    def __calculate_backward_start(tb, L, start=0):
        taob = [start] * L 
        taob[L-1] = start 
        for l in range(L-1)[::-1]:
            taob[l] = taob[l+1] + tb[l+1]
        return taob

    def __calculate_comm_start(tb, taob, sizes, L, P):
        taoc = [0] * L 
        taoc[L-1] = taob[L-1] + tb[L-1]
        tc = [u.predict_allreduce_time_with_size(alpha, beta, s*4, P) for s in sizes]
        for l in range(L-1)[::-1]:
            taoc[l] = max(taoc[l+1] + tc[l+1], taob[l] + tb[l])
        return taoc, tc

    def __merge(taob, tc, p, l, P):
        tc[l] = 0
        p[l-1] = p[l-1]+p[l]
        p[l] = 0
        tc[l-1] = u.predict_allreduce_time_with_size(alpha, beta, p[l-1]*4, P)


    def _algorithm_singlelayer(tb, p, L, P):
        allsize = np.sum(p)
        newp = [0] * L
        newp[0] = allsize
        taob = __calculate_backward_start(tb, L)
        taoc, tc = __calculate_comm_start(tb, taob, newp, L, P)
        return taob, tb, taoc, tc, newp


    def _algorithm_wfbp(tb, p, L, P):
        taob = __calculate_backward_start(tb, L)
        taoc, tc = __calculate_comm_start(tb, taob, p, L, P)
        return taob, tb, taoc, tc, p

    def _algorithm_mgwfbp(tb, p, L, P):
        p = p[:]
        taob = __calculate_backward_start(tb, L)
        taoc, tc = __calculate_comm_start(tb, taob, p, L, P)
        for l in range(1, L-1)[::-1]:
            current_taob = taob[l-1] + tb[l-1]
            if current_taob < taoc[l+1]+tc[l+1]:
                __merge(taob, tc, p, l, P)
                taoc, tc = __calculate_comm_start(tb, taob, p, L, P)
            elif current_taob > taoc[l+1]+tc[l+1] and current_taob < taoc[l]+tc[l] and taoc[l]+alpha > current_taob:
                __merge(taob, tc, p, l, P)
                taoc, tc = __calculate_comm_start(tb, taob, p, L, P)
            else:
                pass
        return taob, tb, taoc, tc, p


    print('layerwise_sizes: ', layerwise_sizes)
    print('layerwise_compute_times: ', layerwise_compute_times)
    p = layerwise_sizes[:]
    tb = layerwise_compute_times[:]
    L = len(p)
    sl_taob, sl_tb, sl_taoc, sl_tc, sl_p= _algorithm_singlelayer(tb, p, L, P)
    print('signlelayer: ', np.sum(p), np.count_nonzero(sl_p), np.sum(sl_tb), np.sum(sl_tc), sl_tc[0]+sl_taoc[0])
    wfbp_taob, wfbp_tb, wfbp_taoc, wfbp_tc, wfbp_p = _algorithm_wfbp(tb, p, L, P)
    print('wfbp: ', np.sum(p), np.count_nonzero(wfbp_p), np.sum(wfbp_tb), np.sum(wfbp_tc), wfbp_tc[0]+wfbp_taoc[0])
    mgwfbp_taob, mgwfbp_tb, mgwfbp_taoc, mgwfbp_tc, mgwfbp_p = _algorithm_mgwfbp(tb, p, L, P)
    print('mgwfbp: ', np.sum(p), np.count_nonzero(mgwfbp_p), np.sum(mgwfbp_tb), np.sum(mgwfbp_tc), mgwfbp_tc[0]+mgwfbp_taoc[0])

if __name__ == '__main__':
    #topk_measured()
    #plot_compute_vs_communication()
    #plot_merged_topk()
    #mgs_simulation()
    mgwfbp_simulation()
    plt.show()

from __future__ import print_function
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
from decimal import Decimal
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from utils import read_log, plot_hist, update_fontsize, autolabel, read_p100_log
from plot_sth import Bar
import os
import plot_sth as Color
import math

OUTPUT_PATH = '/home/shshi/gpuhome/plots/tpds'
#OUTPUT_PATH = '/home/shaohuais/plots/tpds'
INPUT_PATH = './logs'
FONTSIZE=14

#num_of_nodes = [2, 4, 8, 16, 32]
num_of_nodes = [4]
#num_of_nodes = [8, 80, 81, 82, 83, 85]
#num_of_nodes = [16, 32, 64]
B = 9.0 * 1024 * 1024 * 1024.0 / 8 # 10 Gbps Ethernet
#B = 56 * 1024 * 1024 * 1024.0 / 8 # 56 Gbps IB
markers = {2:'o',
        4:'x',
        8:'^'}
formats={2:'-', 4:'-.', 8:':', 16:'--', 32:'-*', 64: '-+'}
gmarkers = {'dense':'o',
        'sparse':'x',
        'topk':'x',
        'gtopk':'^'}
gcolors = {'dense':'b',
        'sparse':'r',
        'topk':'r',
        'gtopk':'g'}

def time_of_allreduce(n, M, B=B):
    """
    n: number of nodes
    M: size of message
    B: bandwidth of link
    """
    # Model 1, TernGrad, NIPS2017
    #if True:
    #    ncost = 100 * 1e-6
    #    nwd = B
    #    return ncost * np.log2(n) + M / nwd * np.log2(n) 

    # Model 2, Lower bound, E. Chan, et al., 2007
    if True:
        #alpha = 7.2*1e-6 #Yang 2017, SC17, Scaling Deep Learning on GPU and Knights Landing clusters
        #alpha = 6.25*1e-6*n # From the data gpuhome benchmark
        #alpha = 12*1e-6*n # From the data gpuhome benchmark
        alpha = 45.25*1e-6#*np.log2(n) # From the data gpuhome benchmark
        beta =  1 / B *1.2
        gamma = 1.0 / (16.0 * 1e9  * 4) * 160
        M = 4*M
        #t = 2*(n)*alpha + 2*(n-1)*M*beta/n + (n-1)*M*gamma/n
        t = 2*(n-1)*alpha + 2*(n-1)*M*beta/n + (n-1)*M*gamma/n
        return t * 1e6
    ts = 7.5/ (1000.0 * 1000)# startup time in second
    #seconds = (np.ceil(np.log2(n)) + n - 1) * ts + (2*n - 1 + n-1) * M / n * 1/B 
    #seconds = (np.ceil(np.log2(n)) + n - 1) * ts + 2 * (n - 1) * 2*M/n * 1/B
    #tcompute = 1. / (2.2 * 1000 * 1000 * 1000)
    tcompute = 1. / (1 * 1000 * 1000 * 1000)
    #seconds = 2 * (n - 1) * ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute
    #C = 1024.0 * 1024 # segmented_size 
    #if M > C * n:
    #    # ring_segmented allreduce
    #    seconds = (M / C + (n - 2)) * (ts + C / B + C * tcompute)
    #else:
        # ring allreduce, better than the above
        #seconds = (n - 1) * ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute
    seconds = 2*(n-1)*n*ts + 2 * (n - 1) * M/n * 1/B + (n-1)*M/n * tcompute

    #C =  512.0
    #seconds = (M / C + n-2) * (ts + C/B)
    return seconds * 1000 * 1000 # micro seconds



class Simulator():
    def __init__(self, name, computes, sizes, num_of_nodes, render=True):
        self.name = name
        self.computes = computes
        self.sizes = sizes
        self.num_of_nodes = num_of_nodes
        self.comms = None
        self.title = name + ' (WFBP)'
        self.max_time = 0
        self.ax = None
        self.render = render
        self.merged_layers = []

    def wfbp(self, with_optimal=False):
        start_time = 0.0
        comm_start_time = 0.0
        comm = 0.0
        if not self.comms:
            comms = [time_of_allreduce(self.num_of_nodes, s, B) for s in self.sizes]
        else:
            comms = self.comms
        max_time = max(np.sum(self.computes), np.sum(comms)+self.computes[0])
        print('Layer-wise total comm. time:', np.sum(comms)/1000.)
        if not with_optimal:
            self.max_time = max_time
        if not self.ax and self.render:
            fig, ax = plt.subplots(1, figsize=(30, 3))
            #ax.set_title(self.title, x=0.5, y=0.8)
            self.ax = ax
        comm_layer_id = ''
        for i in range(len(self.computes)):
            comp = self.computes[i]
            layer_id = len(self.computes) - i
            if not with_optimal:
                if self.render:
                    bar = Bar(start_time, comp, self.max_time, self.ax, type='p', index=layer_id)
                    bar.render()
            if comm_start_time + comm > start_time + comp:
                comm_start_time = comm_start_time + comm
            else:
                comm_start_time = start_time + comp
            if comm == 0.0 and comm_layer_id != '':
                comm_layer_id = str(comm_layer_id)+','+str((len(self.computes) - i))
            else:
                comm_layer_id = str(layer_id)

            comm = comms[i]
            type = 'wc'
            if with_optimal:
                type = 'mc'
            if self.render:
                bar_m = Bar(comm_start_time, comm, self.max_time, self.ax, type=type, index=comm_layer_id, is_optimal=with_optimal)
                bar_m.render()
            start_time += comp 
        total_time = (comm_start_time + comm)/1000.0
        title='MG-WFBP' if with_optimal else 'WFBP'
        print(title+' Total time: ', total_time, ' ms')
        if self.render:
            plt.subplots_adjust(left=0.06, right=1.)
        return total_time

    def synceasgd(self):
        start_time = 0.0
        comm_start_time = 0.0
        comm = 0.0
        total_size = np.sum(self.sizes)
        comm = time_of_allreduce(self.num_of_nodes, total_size, B)
        total_comp = np.sum(self.computes)
        comm_start_time = total_comp
        index = ','.join([str(len(self.computes)-i) for i in range(0, len(self.computes))])
        if self.render:
            bar = Bar(np.sum(self.computes), comm, self.max_time, self.ax, type='sc', index=index)
            bar.render()
        total_time = (comm_start_time + comm)/1000.0
        print('SyncEASGD Total time: ', total_time, ' ms')
        if self.render:
            pass
        return total_time



    def cal_comm_starts(self, comms, comps):
        """
        comms and comps have been aligned
        """
        start_comms = []
        start_comms.append(0.0)
        sum_comp = 0.0
        for i in range(1, len(comms)):
            comm = comms[i-1]
            comp = comps[i-1]
            #print(start_comms[i-1],comm, sum_comp,comp)
            start_comm = max(start_comms[i-1]+comm, sum_comp+comp)
            #print('start_comm: ', start_comm, ', comm: ', comm)
            start_comms.append(start_comm)
            sum_comp += comp
        return start_comms

    def merge(self, comms, sizes, i, p, merge_size, comps):
        comms[i] = 0# merge here
        comms[i+1] = p
        sizes[i+1] = merge_size 
        start_comms = self.cal_comm_starts(comms, comps)
        #print('start_comms: ', start_comms)
        self.merged_layers.append(i)
        return start_comms

    def gmwfbp2(self):
        if not self.comms:
            comms = [time_of_allreduce(self.num_of_nodes, s, B) for s in self.sizes]
        else:
            comms = self.comms

        #comms = comms[0:-1]
        #print('comms: ', comms)
        comps = self.computes[1:]
        comps.append(0) # for last communication

        optimal_comms = list(comms)
        optimal_sizes = list(self.sizes)
        start_comms = self.cal_comm_starts(optimal_comms, comps)
        sum_comp = 0.0
        #print('start_comms: ', start_comms)
        #return

        for i in range(0, len(comms)-1):
            comp = comps[i]
            comm = optimal_comms[i]
            if start_comms[i] + comm > comp+sum_comp:
                # cannot be hidden, so we need to merge
                merge_size = optimal_sizes[i+1] + optimal_sizes[i]
                r = comm + optimal_comms[i+1]
                p = time_of_allreduce(self.num_of_nodes, merge_size, B) 
                if start_comms[i] >= comp+sum_comp:
                    # don't care about computation
                    if p < r:
                        start_comms = self.merge(optimal_comms, optimal_sizes, i, p, merge_size, comps)
                        #optimal_comms[i] = 0# merge here
                        #optimal_comms[i+1] = p
                        #optimal_sizes[i+1] += merge_size 
                        #start_comms = self.cal_comm_starts(optimal_comms, comps)
                else:
                    if comp+sum_comp+p < start_comms[i]+comm+optimal_comms[i+1]:
                        start_comms = self.merge(optimal_comms, optimal_sizes, i, p, merge_size, comps)
            else:
                pass # optimal, nothing to do
            sum_comp += comp
        optimal_comms.append(comms[-1])
        self.wfbp()
        self.synceasgd()
        self.comms = optimal_comms
        self.title = self.name+ ' (GM-WFBP)'
        ret = self.wfbp(with_optimal=True)
        #print('merged-layers: ', self.merged_layers)
        return ret

#start = 1024*256
#end = 1024*512
#start = 1024*513
#start = 0
#end = 1024*1024*128
def read_times_from_nccl_log(logfile, mode='allreduce', start=0, end=128*1024*1024):
    print('fn: ', logfile)
    f = open(logfile)
    sizes = []
    times = []
    size_comms = {}
    for line in f.readlines():
        items = ' '.join(line.split()).split(' ')
        if (len(items) == 11 or len(items) == 12) and items[0] != '#':
            try:
                size = int(items[0])
            except:
                continue
            if size == 8:
                continue
            if (size >= start and size <= end):
                if size not in size_comms:
                    size_comms[size] = [] 
                try:
                    if mode == 'allreduce':
                        t = float(items[4])/(1000*1000)
                    else:
                        t = float(items[3])/(1000*1000)
                    size_comms[size].append(t)
                    sizes.append(size)
                    times.append(t)
                except:
                    continue
            #print(items)
    f.close()
    sizes = size_comms.keys()
    sizes.sort()
    comms = [np.mean(size_comms[s]) for s in sizes]
    errors = [np.std(size_comms[s]) for s in sizes]
    #print('sizes: ', sizes)
    #print('errors: ', errors)
    return np.array(sizes), np.array(comms), np.array(errors)

def read_times_from_horovod_log(logfile):
    print('fn: ', logfile)
    f = open(logfile)
    sizes = []
    times = []
    size_comms = {}
    for line in f.readlines():
        items = line.split(' ')
        try:
            size = int(items[0])
        except:
            continue
        t = float(items[1][:-1])
        if size not in size_comms:
            size_comms[size] = []
        size_comms[size].append(t)
    f.close()
    sizes = size_comms.keys()
    sizes.sort()
    comms = [np.mean(size_comms[s]) for s in sizes]
    errors = [np.std(size_comms[s]) for s in sizes]
    #print('sizes: ', sizes)
    #print('errors: ', errors)
    return np.array(sizes), np.array(comms), np.array(errors)


def read_allreduce_log(filename, start=2e5, end=5e5):
    print('filename: ', filename)
    f = open(filename, 'r')
    sizes = []
    comms = []
    size_comms = {}
    for l in f.readlines():
        if l[0] == '#' or l[0] == '[' or len(l)<10 :
            continue
        items = ' '.join(l.split()).split()
        comm = float(items[-1])/1.e6
        size = int(items[0])#/4

        if size > end or size < start:
            continue
        comms.append(comm)
        sizes.append(size)
        if size not in size_comms:
            size_comms[size] = []
        size_comms[size].append(comm)
    f.close()
    sizes = size_comms.keys()
    sizes.sort()
    #print('sizes: ', sizes)
    comms = [np.mean(size_comms[s]) for s in sizes]
    errors = [np.std(size_comms[s]) for s in sizes]
    return np.array(sizes), np.array(comms), np.array(errors)


def predict(filename, n, color, marker, label, sizes=None, ax=None, nccl=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4.5))
    if sizes is None:
        if not nccl:
            sizes, comms, errors = read_allreduce_log(filename)
            label='%d nodes' % (n)
        else:
            mode = 'allgather'
            sizes, comms, errors = read_times_from_nccl_log(filename, mode=mode)
            label='%d GPUs' % (n)
        size_in_kbytes = np.array(sizes) #/ 1024
        #print('size_in_kbytes: ', size_in_kbytes)
        #print('comms: ', comms)
        #plt.plot(size_in_kbytes, comms, c=color, marker=marker, label=label+' measured', linewidth=2)
        #plt.scatter(size_in_kbytes, comms, label=label+' measured', linewidth=2)
        ax.errorbar(size_in_kbytes, comms, errors, fmt=formats[n], label=label, linewidth=1)
        #plt.plot(sizes, comms, c=color, marker=marker, label=label, linewidth=2)
    #bandwidths = np.array(sizes)/np.array(comms)
    #plt.plot(sizes, bandwidths, c=color, marker=marker, label=label, linewidth=2)
    predicts = []
    for M in sizes:
       p = time_of_allreduce(n, M, B) 
       predicts.append(p)
    #rerror = (np.array(predicts)-np.array(comms))/np.array(comms)
    #print('erro: ', np.mean(np.abs(rerror)))
    #plt.scatter(sizes, predicts, c='red', marker=markers[n])
    #jax.plot(size_in_kbytes, predicts, c=color, marker=marker, linestyle='--', label=label+' predict', markerfacecolor='white', linewidth=1)
    return sizes

def plot_all_communication_overheads():
    #labels = ['2-node', '4-node', '8-node', '16-node']
    fig, ax = plt.subplots(figsize=(5,4.5))
    labels = ['%d-node' % i for i in num_of_nodes]
    colors = ['r', 'g', 'b', 'black', 'y', 'c']
    markers = ['^', 'o', 'd', '*', 'x', 'v']
    sizes = None
    #sizes = np.arange(128.0, 1e5, step=8192)
    for i, n in enumerate(num_of_nodes):
        #test_file = '%s/mgdlogs/mgd140/ring-allreduce%d.log' % (INPUT_PATH, n)
        test_file = '%s/nccl/ring-allreduce-k80-%d.log' % (INPUT_PATH, n)
        print('logfile: ', test_file)
        predict(test_file, n, colors[i], markers[i], labels[i], sizes, ax, True)
        
    plt.xlabel('Size of parameters (Bytes)')
    plt.ylabel(r'Communication time ($\mu$s)')
    plt.ylim(bottom=0, top=plt.ylim()[1]*1.2)
    plt.xlim(left=0)
    plt.legend(ncol=1, loc=2, prop={'size': 10})
    update_fontsize(ax, fontsize=14)
    plt.subplots_adjust(left=0.18, bottom=0.13, top=0.91, right=0.92)
    #plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'commtime'))
    plt.show()

def gmwfbp_simulate():
    name = 'GoogleNet'
    #name = 'ResNet'
    #name = 'VGG'
    #name = 'DenseNet'
    num_of_nodes = 32
    test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8comm.log' % name.lower()
    sizes, comms, computes, merged_comms = read_log(test_file)
    #computes = [c/4 for c in computes]
    #sizes = [1., 1., 1., 1.]
    #computes = [3., 3.5, 5., 6.]
    #sim = Simulator(name, computes[0:4], sizes[0:4], num_of_nodes)
    sim = Simulator(name, computes, sizes, num_of_nodes)
    #sim.wfbp()
    sim.gmwfbp2()
    plt.savefig('%s/breakdown%s.pdf' % (OUTPUT_PATH, name.lower()))
    #plt.show()

def gmwfbp_speedup():
    #configs = ['GoogleNet', 64]
    configs = ['ResNet', 32]
    #configs = ['DenseNet', 128]
    name = configs[0] 
    b = configs[1]
    test_file = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8comm.log' % name.lower()
    sizes, comms, computes, merged_comms = read_log(test_file)
    device = 'k80'

    #device = 'p100'
    #pfn = '/media/sf_Shared_Data/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/tmp8commp100%s.log' % (name.lower(), name.lower())
    #val_sizes, computes = read_p100_log(pfn)
    #print('computes: ', np.sum(computes))
    #print('computes: ', computes)
    #assert len(computes) == len(sizes)

    nnodes = [4, 8, 16, 32, 64]
    #nnodes = [2, 4, 8]
    wfbps = []
    gmwfbps = []
    synceasgds = []
    micomputes = np.array(computes)
    tf = np.sum(micomputes) * 0.5 / 1000
    tb = np.sum(micomputes) / 1000
    total_size = np.sum(sizes)
    single = b/(tf+tb)
    optimal = []
    colors = ['k', 'r', 'g', 'b']
    markers = ['s', '^', 'o', 'd']
    for num_of_nodes in nnodes:
        sim = Simulator(name, computes, sizes, num_of_nodes, render=False)
        wfbp = sim.wfbp()
        wfbps.append(b*num_of_nodes/(wfbp+tf)/single)
        gmwfbp = sim.gmwfbp2()
        gmwfbps.append(b*num_of_nodes/(gmwfbp+tf)/single)
        tc = time_of_allreduce(num_of_nodes, total_size, B)/1000
        print('#nodes:', num_of_nodes, ', tc: ', tc) 
        synceasgd = tb + tf + tc
        synceasgds.append(b*num_of_nodes/synceasgd/single)
        optimal.append(num_of_nodes)
    print('tf: ', tf)
    print('tb: ', tb) 
    print('total_size: ', total_size)
    print('wfbp: ', wfbps)
    print('gmwfbps: ', gmwfbps)
    print('synceasgds: ', synceasgds)
    print('compared to synceasgds: ', np.array(gmwfbps)/np.array(synceasgds))
    print('compared to wfbps: ', np.array(gmwfbps)/np.array(wfbps))
    fig, ax = plt.subplots(figsize=(5,4.5))
    ax.plot(nnodes, optimal, color='k', marker='s', label='Linear')
    ax.plot(nnodes, wfbps, color='r', marker='d', label='WFBP')
    ax.plot(nnodes, synceasgds, color='b', marker='o', label='SyncEASGD')
    ax.plot(nnodes, gmwfbps, color='g', marker='^', label='MG-WFBP')
    plt.legend(loc=2)
    plt.xlabel('# of nodes')
    plt.ylabel('Speedup')
    #plt.title('%s-Simulation'%name)
    #plt.yscale('log', basey=2)
    #plt.xscale('log', basey=2)
    plt.ylim(bottom=1,top=nnodes[-1]+1)
    plt.xlim(left=1, right=nnodes[-1]+1)
    plt.xticks(nnodes)
    plt.yticks(nnodes)
    plt.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    update_fontsize(ax, fontsize=14)
    plt.subplots_adjust(left=0.13, bottom=0.13, top=0.96, right=0.97)
    #plt.savefig('%s/speedup%s.pdf' % (OUTPUT_PATH, name.lower()+device))
    plt.show()




def realdata_speedup():
    nworkers = [1, 4, 8, 16, 32]
    configs = ['VGG-16', 128]
    dense=  [1317.333, 104.200, 92.560 , 39.480  ,12.600]
    topk=   [1317.333, 110.576, 109.900, 97.865  ,63.002]
    gtopk=  [1317.333, 131.060, 130.551, 126.434 ,123.200]

    #configs = ['ResNet-20', 32]
    #dense=  [920.632, 821.700, 705.200, 520.400, 287.900]
    #topk=   [920.632, 908.837, 752.985, 737.594, 696.029]
    #gtopk=  [920.632, 916.260, 868.730, 808.500, 789.300]

    #configs = ['AlexNet', 32]
    #dense = [173.469, 14.010,  12.118,  4.936 ,  1.234]
    #topk = [173.469, 14.238,  13.865,  13.352,  9.236]
    #gtopk = [173.469, 16.536,  16.446,  16.359,  15.777]

    #configs = ['ResNet-50', 32]
    #dense =[52.873,  39.002,  36.989,  23.176,  10.721]
    #topk = [52.873,  37.729,  35.703,  34.495,  30.583]
    #gtopk =[52.873,  39.795,  39.713,  39.060,  39.119]

    configs = ['LSTM-PTB', 32]
    dense =[392.0,  12.657,  8.7,  4.1, 2.1]
    topk = [392.0,  19.9,  18.6,  14.8,  5.4]
    gtopk =[392.0,  17.8,  17.6,  15.1,  10.8]

    name = configs[0] 
    fig, ax = plt.subplots(figsize=(5,4))
    optimal = [100 for i in range(len(dense)-1)] 

    dense = [v/dense[0]*100 for i, v in enumerate(dense[1:])]
    topk = [v/topk[0]*100 for i, v in enumerate(topk[1:])]
    gtopk = [v/gtopk[0]*100 for i, v in enumerate(gtopk[1:])]
    todense = np.array(gtopk)/np.array(dense)
    totopk= np.array(gtopk)/np.array(topk)
    print(name, ', compared to dense: ', todense, 'mean: ', np.mean(todense))
    print(name, ', compared to topk: ', totopk, 'mean: ', np.mean(totopk))
    #ax.plot(nworkers[1:], optimal, color='k', marker='s', label='Optimal')
    ax.plot(nworkers[1:], dense, color=gcolors['dense'], marker=gmarkers['dense'], label='Dense S-SGD')
    ax.plot(nworkers[1:], topk, color=gcolors['topk'], marker=gmarkers['topk'], label=r'Top-$k$ S-SGD')
    ax.plot(nworkers[1:], gtopk, color=gcolors['gtopk'], marker=gmarkers['gtopk'], label=r'gTop-$k$ S-SGD')
    #plt.yscale('log', basey=2)
    #plt.xscale('log', basey=2)
    plt.legend(loc=3,prop={'size': 14})
    plt.xlabel('# of workers (GPU)')
    plt.ylabel('Scaling efficiency (Percentage)')
    plt.xticks(nworkers[1:])
    plt.title(name)
    #plt.yticks(nnodes)
    #plt.ylim(top=gtopk[-1]+1)
    #plt.xlim(left=1, right=nnodes[-1]+1)
    #plt.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    plt.grid(linestyle=':')
    update_fontsize(ax, fontsize=14)
    plt.subplots_adjust(left=0.18, bottom=0.16, top=0.92, right=0.97)
    plt.savefig('%s/scalingeffi%s.pdf' % (OUTPUT_PATH, name.lower()))
    plt.show()

def parse_real_comm_cost():
    configs = ['GoogleNet', 'gm'] #SyncEASGD
    name = configs[0]
    t = configs[1] 
    nnodes = [2, 4, 8]
    ncomms = []
    for n in nnodes:
        test_file = '/home/shshi/gpuhome/repositories/dpBenchmark/tools/caffe/cnn/%s/%s%dcomm.log' % (name.lower(), t, n)
        sizes, comms, computes, merged_comms = read_log(test_file)
        ncomms.append(np.sum(merged_comms))
    print('network: ', name, ', type: ', t)
    print('ncomms: ', ncomms)


def speedup_with_r_and_n(r, n):
    return n/(1.+r)

def draw_ssgd_speedup():
    Ns = [8, 16, 32, 64]
    r = np.arange(0, 4, step=0.1)
    for N in Ns:
        s = N / (1+r)
        plt.plot(r, s)
    #plt.yscale('log', basey=2)
    plt.show()

def plot_p2platency(nccl=True):
    def _fit_linear_function(x, y):
        X = np.array(x).reshape((-1, 1))
        Y = np.array(y)
        #print('x: ', X)
        #print('y: ', Y)
        model = LinearRegression()
        model.fit(X, Y)
        alpha = model.intercept_
        beta = model.coef_[0]
        #A = np.vstack([X, np.ones(len(X))]).T
        #beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
        return alpha, beta
    fig, ax = plt.subplots(figsize=(5,3.5))
    #fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=(5,4.2))
    #mode='allgather'
    mode='allreduce'
    #msg_type='large' # 'small' or 'large'
    msg_type='small-v6' # 'small' or 'large'
    connection = '10GbE'
   # msg_type='small2' # 'small' or 'large'
   # connection = '56GbIB'
    nworkers=16
    #gpu='V100';nccl=True;horovod=False
    gpu='GK210';nccl=False;horovod=False
    #gpu='V100';nccl=True;horovod=True
    if nccl:
        if horovod:
            filename = './logs/v100-%s-horovod-nccl-%s-n%d.log' % (connection, msg_type, nworkers)
        else:
            filename = './logs/v100-%s-%s-nccl-%s-%d.log' % (connection, mode, msg_type, nworkers)
    else:
        filename = '/home/shshi/gpuhome/repositories/mpibench/allreduce%d.log' % 8 

    if nccl:
        if horovod:
            sizes, comms, errors = read_times_from_horovod_log(filename)
        else:
            sizes, comms, errors = read_times_from_nccl_log(filename, mode=mode)
    else:
        sizes, comms, errors = read_allreduce_log(filename)
    alpha, beta = _fit_linear_function(sizes, comms)
    print('alpha: ', alpha, ', beta: ', beta)

    #ax.errorbar(sizes, comms, errors, label='Measured', linewidth=0.5)
    #sizes = sizes[comms<0.02]
    #comms = comms[comms<0.02]
    print('sizes: ', sizes)
    ax.plot(sizes, comms*1e6, label='16-%s (%s) Measured' % (gpu, connection), linewidth=0.5)
    ax.plot(sizes, (alpha+np.array(sizes)*beta)*1e6, label=r'Fitted ($a$=%.2e,$b$=%.2e)'%(alpha, beta), linewidth=1, color='r', linestyle='--')
    ax.grid(linestyle=':')
        
    plt.xlabel('Size of parameters [bytes]')
    plt.ylabel(r'Communication time [$\mu s$]')
    #plt.ylim(bottom=0, top=plt.ylim()[1]*1.2)
    #plt.xlim(left=0)
    plt.legend(ncol=1, loc=4, prop={'size': 12})
    update_fontsize(ax, fontsize=12)
    #plt.subplots_adjust(left=0.16, bottom=0.17, top=0.98, right=0.98)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    #plt.subplots_adjust(left=0.21, bottom=0.13, top=0.96, right=0.97)
    plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'commtime-%d-%s-%s'%(nworkers,gpu, connection)), bbox_inches='tight')
    plt.show()

def plot_allreduce_comparison():
    alpha = 0.436
    beta =  4*9e-6

    def _denseallreduce_model(P, m):
        return 2*(P-1)*alpha + 2* (P-1)/P * m * beta
        #return 2*np.log2(P)*alpha + 2* (P-1)/P * m * beta

    def _sparseallreduce_model(P, m, rho=0.001):
        return np.log2(P) + 2 * (P - 1) * rho * m * beta

    def _gtopkallreduce_model(P, m, rho=0.001):
        return 2*np.log2(P) + 4 * np.log2(P) * rho * m * beta

    fig, ax = plt.subplots(figsize=(5,3.8))
    #fig, ax = plt.subplots(figsize=(5,4.2))

    variable = 'm'
    variable = 'P'
    variable = 'rho'
    if variable == 'm':
        #m = [2**(2*10+i) for i in range(0, 8)] # from 1M to 128M
        m = [2**(10+i) for i in range(0, 10)] # from 1K to 1M 
        m = np.array(m)
        P = 32 
        rho = 0.001
        #xlabel = 'Size of parameters [bytes]'
        xlabel = '# of parameters'
        xticks = m
        # measured
        #filename = '%s/mgdlogs/mgd140/ring-allreduce%d.log' % (INPUT_PATH, P)
        #sizes, comms, errors = read_allreduce_log(filename)
        #comms = np.array(comms)/1000.
        #print('sizes: ', sizes)
        #print('comms: ', comms)
        #ax.plot(sizes, comms, label=r'DenseAllReduce', linewidth=1, marker=gmarkers['dense'], color=gcolors['dense'])
    elif variable == 'P':
        m = 25*1024 * 1024 # 10MBytes
        P = np.array([4, 8, 16, 32, 64, 128])
        rho = 0.001
        xlabel = 'Number of workers'
        xticks = P
    elif variable == 'rho':
        #m = 8*1024 * 1024 # 10MBytes
        m = 1*1024 * 1024 # 1MBytes
        P = 32 #np.array([4, 8, 16, 32])
        #rho = np.array([0.1/(2*i) for i in range(1, 10)])
        rho = np.arange(0.001, 0.05, step=0.001)
        xlabel = 'Density'
        xticks = rho

    dar = _denseallreduce_model(P, m)
    if variable == 'rho':
        dars = [dar for i in range(len(rho))]
        dar = dars
    sar = _sparseallreduce_model(P, m, rho)
    gar = _gtopkallreduce_model(P, m, rho)
    
    ax.plot(xticks, dar, label=r'DenseAllReduce', linewidth=1, marker=gmarkers['dense'], color=gcolors['dense'])
    ax.plot(xticks, sar, label=r'TopKAllReduce', linewidth=1, marker=gmarkers['sparse'], color=gcolors['sparse'])
    #ax.plot(xticks, gar, label=r'gTopKAllReduce', linewidth=1, marker=gmarkers['gtopk'], color=gcolors['gtopk'])


    ax.grid(linestyle=':')
    plt.subplots_adjust(bottom=0.16, left=0.15, right=0.96, top=0.97)
    #ax.set_yscale("log", nonposy='clip')
    plt.xlabel(xlabel)
    plt.ylabel(r'Communication time [ms]')
    #plt.ylim(bottom=0, top=plt.ylim()[1]*1.2)
    plt.legend(ncol=1, loc=2, prop={'size': 10})
    plt.subplots_adjust(left=0.18, bottom=0.20, top=0.94, right=0.96)
    #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    if variable == 'P':
        plt.xticks(xticks)
    elif variable == 'm':
        ax.set_xscale("log")
    update_fontsize(ax, fontsize=16)
    #plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'sparvsgtopk_dynamic%s'%variable))
    plt.show()


def plot_breakdown():
    logpath='/media/sf_Shared_Data/tmp/icdcs2019/mgdlogs/mgd115-2/logs/allreduce-comp-baseline-gwarmup-dc1-modelmgd-speed/'
    networks=['vgg16', 'resnet20', 'alexnet', 'resnet50']
    batchsizes=[128, 128, 64, 256]
    lrs=[0.1, 0.1, 0.01, 0.01]
    nss=[1,1,1, 16]
    for i, na in enumerate(networks):
        bs = batchsizes[i]
        lr = lrs[i]
        ns = nss[i]
        fn = os.path.join(logpath, '%s-n32-bs%d-lr%.4f-ns%d-sg2.50/MGD-0.log' % (na, bs, lr, ns))
        print('fn: ', fn)
    names = ['Compu.', 'Compr.', 'Commu.']
    vgg16=[0.139536,    0.091353,    0.811753]
    resnet20=[0.146005,    0.001618,    0.024686]
    alexnet=[0.257205,    0.383776,    3.36298]
    resnet50=[4.882041,    0.15405 ,  1.424253]
    ratio_vgg16 = [v/np.sum(vgg16) for v in vgg16]
    ratio_resnet20= [v/np.sum(resnet20) for v in resnet20]
    ratio_alexnet = [v/np.sum(alexnet) for v in alexnet]
    ratio_resnet50= [v/np.sum(resnet50) for v in resnet50]
    datas = [ratio_vgg16, ratio_resnet20, ratio_alexnet, ratio_resnet50]
    for d in datas:
        print('ratios: ', d)
    communications = [ratio_vgg16[2], ratio_resnet20[2], ratio_alexnet[2], ratio_resnet50[2]]
    compressions = [ratio_vgg16[1], ratio_resnet20[1], ratio_alexnet[1], ratio_resnet50[1]]
    computes = [ratio_vgg16[0], ratio_resnet20[0], ratio_alexnet[0], ratio_resnet50[0]]
    computes = np.array(computes)
    compressions= np.array(compressions)
    communications= np.array(communications)
    fig, ax = plt.subplots(figsize=(4.8,3.4))

    count = len(datas)
    ind = np.arange(count)
    width = 0.35
    margin = 0.05
    xticklabels = ['VGG-16', 'ResNet-20', 'AlexNet', 'ResNet-50']
    #ind = np.array([s+i+1 for i in range(count)])
    newind = np.arange(count)
    p1 = ax.bar(newind, computes, width, color=Color.comp_color,hatch='x', label=names[0])
    p2 = ax.bar(newind, compressions, width, bottom=computes, color=Color.compression_color,hatch='-', label=names[1])
    p3 = ax.bar(newind, communications, width, bottom=computes+compressions, color=Color.opt_comm_color,label=names[2])

    ax.text(10, 10, 'ehhlo', color='b')
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend([handles[0][0]], [labels[0][0]], ncol=2)
    print(labels)
    print(handles)
    #ax.set_xlim(left=1+0.3)
    #ax.set_ylim(top=ax.get_ylim()[1]*1.3)
    ax.set_xticks(ind)
    ax.set_xticklabels(xticklabels)
    #ax.set_xlabel('Model')
    ax.set_ylabel('Percentage')
    update_fontsize(ax, 10)
    ax.legend((p1[0], p2[0], p3[0]), tuple(names), ncol=9, bbox_to_anchor=(1, -0.1))#, handletextpad=0.2, columnspacing =1.)
    #ax.legend((p1[0], p2[0]), (labels[0],labels[1] ), ncol=2, handletextpad=0.2, columnspacing =1.)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.19, top=0.94)
    plt.savefig('%s/breakdown.pdf' % (OUTPUT_PATH))
    plt.show()


def plot_realdata_comm():
    fig, ax = plt.subplots(figsize=(4.8,3.4))
    #connection = '10GbE'
    connection = '56GbIB'

    configs = ['ResNet-152', 128]
    computation_fp32 = 0.883118444
    computation_fp16 = 0.388587889
    eth_fp32 = np.array([1.23343938852773,1.36014631802352,1.12611348980156])-computation_fp32
    eth_fp16 = np.array([0.757144753582024,0.636332837191469,0.62969512705326])-computation_fp16
    ib_fp32 = np.array([1.08207490737751,1.14349856500806,1.08926123052581])-computation_fp32
    ib_fp16 = np.array([0.531046268746226,0.518222719609269,0.486271295133516])-computation_fp16

    #configs = ['Inception-v4', 128]
    #computation_fp32 = 0.61082206 
    #computation_fp16 = 0.279730928
    #eth_fp32 = np.array([0.943730968358381, 0.899976457881453, 0.780575902070508])-computation_fp32
    #eth_fp16 = np.array([0.591019579388972, 0.477115832605121, 0.423175552678714])-computation_fp16
    #ib_fp32 = np.array([0.767364986034143,0.746227280857805,0.661337401153554])-computation_fp32
    #ib_fp16 = np.array([0.430170979831572,0.373202685750795,0.363820604043981])-computation_fp16
    
    #configs = ['DenseNet-161', 64]
    #computation_fp32 = 0.47765481 
    #computation_fp16 = 0.253018587 
    #eth_fp32 = np.array([0.83238836059332, 0.682584589864119, 0.628013485914495])-computation_fp32
    #eth_fp16 = np.array([0.614905695164206,0.39935562610809,0.360415335875275])-computation_fp16
    #ib_fp32 = np.array([0.601903116710871,0.596814522380611,0.527390124793952])-computation_fp32
    #ib_fp16 = np.array([0.400008289616235,0.329964755038104,0.321050630321988])-computation_fp16

    #configs = ['DenseNet-201', 64]
    #computation_fp32 = 0.360877356 
    #computation_fp16 = 0.215181737 
    #eth_fp32 = np.array([0.7636,0.51283,0.4566])-computation_fp32
    #eth_fp16 = np.array([0.614204217901379,0.406464315896478,0.361810340425332])-computation_fp16
    #ib_fp32 = np.array([0.512376011576687,0.437788525915094,0.405942742968561])-computation_fp32
    #ib_fp16 = np.array([0.397707045163982,0.354400110781044,0.331319474344984])-computation_fp16

    if connection == '10GbE':
        fp32 = eth_fp32
        fp16 = eth_fp16
    elif connection == '56GbIB':
        fp32 = ib_fp32
        fp16 = ib_fp16
    comps = [computation_fp32, computation_fp16]
    datas = np.array([fp32, fp16]).transpose()
    print('data shape: ', datas.shape)
    count = len(comps)
    width = 0.23
    margin = 0.02
    xticklabels = ['FP32', 'FP16']
    s = (1 - (width*count+(count-1) *margin))/2+2.4*width
    ind = np.array([s+i+1 for i in range(count)])
    centerind = None
    labels=['WF.', 'S.E.', 'M.W.']
    for i, l in enumerate(datas):
        data = datas[i]
        comms = data
        newind = ind+s*width+(s+1)*margin
        print('shape: ', newind.shape)
        p1 = ax.bar(newind, comps, width, color=Color.comp_color,hatch='x',edgecolor='black', label='Comp.')
        p2 = ax.bar(newind, comms, width,
                             bottom=comps, color=Color.comm_color,edgecolor='black', label='Comm.')

        s += 1 
        autolabel(p2, ax, labels[i], 0)
        print('comp: ', comps)
        print('comms: ', comms)
        print('')

    rects = ax.patches
    ax.text(10, 10, 'ehhlo', color='b')
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend([handles[0][0]], [labels[0][0]], ncol=2)
    print(labels)
    print(handles)
    #ax.set_xlim(left=1+0.3)
    ax.set_ylim(top=ax.get_ylim()[1]*1.3)
    ax.set_xticks(ind+2*(width+margin))
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Training mode')
    ax.set_ylabel('Time [s]')
    update_fontsize(ax, 14)
    ax.legend((p1[0], p2[0]), (labels[0],labels[1] ), ncol=2, handletextpad=0.2, columnspacing =1.)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.17, top=0.94)
    plt.savefig('%s/%s-breakdown-bs%d-%s.pdf' % (OUTPUT_PATH, configs[0].lower(), configs[1], connection))
    #plt.show()



if __name__ == '__main__':
    #plot_all_communication_overheads()
    #plot_p2platency()
    #plot_allreduce_comparison()
    #realdata_speedup()
    plot_realdata_comm()

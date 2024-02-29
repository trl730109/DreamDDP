import matplotlib.pyplot as plt
import os
import glob
import itertools
import numpy as np
import pandas as pd
import numpy as np
import utils
from reader import read_times_from_nccl_log
from decimal import Decimal
from sklearn.linear_model import LinearRegression
import plot_sth as Color

OUTPUT_PATH = '/media/tmp/infocom2021'
FONTSIZE=18
def plot_bw_vs_jobs(ax, nodes, msg_size, job_num, rdma=0):
    job_list = list(range(1, job_num + 1))
    comm_list = [[] for nnode in nodes]
    for i, nnode in enumerate(nodes):
        comms = comm_list[i]
        for n_job in job_list:
            folder='logs/infocom21/nccl_rdma%d_job_nw%d_n%d_s%d' % (rdma, nnode, n_job, msg_size)
            tmp_cs = []
            error = 0
            for k in range(1, n_job+1):
                fn = 'nccl_job_%d.log' % k
                logfile = os.path.join(folder, fn)
                try:
                    _, c, _ = read_times_from_nccl_log(logfile, end=512*1024*1024, original=True)
                    c = c[0]
                except:
                    c = 0.0
                    error += 1
                tmp_cs.append(c)
            print('task[msg_size=%d, #jobs=%d, #node=%d, error: %d' % \
                (msg_size, n_job, nnode, error))
            c = np.max(tmp_cs) # us
            throughput = msg_size*(n_job-error)/1e6 / c
            comms.append(throughput)
    #print('nodes: ', nodes)
    #print('job_list: ', job_list)
    #print('comm_list: ', comm_list)

    MB = float(msg_size)/(1024*1024)
    for i, nnode in enumerate(nodes):
        ax.plot(job_list, comm_list[i], label='# Nodes=%d, Size=%.2fMB'% (nnode, MB))
    ax.set_xlabel("# Job", size=18)
    ax.set_ylabel("Total Throughput [MB/s]", size=18)
    ax.legend()

def plot_multiple_jobs():
    #nodes = list(range(2, 8, step=2))
    #nodes = [2, 4, 6, 8]
    nodes = [32]
    job_num = 2
    #msg_sizes = [2**(11+i) for i in range(8, 17)]
    msg_size1 = list(range(512, 1048576, 10240))
    msg_size2 = [] #list(range(1048576, 4194304, 32768))
    msg_sizes= msg_size1+msg_size2 
    fig, ax = plt.subplots(figsize = (12, 8))
    RDMA=0
    for msg_size in msg_sizes:
        plot_bw_vs_jobs(ax, nodes, msg_size, job_num, RDMA)
    plt.show()

def plot_allreduce_throughput():
    #msg_size1 = list(range(512, 1048576, 10240))
    #msg_size2 = list(range(1048576, 4194304, 32768))
    #msg_sizes= msg_size1+msg_size2 
    fig, ax = plt.subplots()
    iperfbw=9.43
    msg_sizes= list(range(512, 4194304, 32768))
    job_list = [1, 2]
    job_data = [[] for j in job_list]
    nnode=8
    rdma=0
    #fig, ax = plt.subplots(figsize = (12, 8))
    for i, n_job in enumerate(job_list):
        for msg_size in msg_sizes:
            folder='logs/infocom21-v2/nccl_rdma%d_job_nw%d_n%d_s%d' % (rdma, nnode, n_job, msg_size)
            tmp_cs = []
            error = 0
            for k in range(1, n_job+1):
                fn = 'nccl_job_%d.log' % k
                logfile = os.path.join(folder, fn)
                try:
                    _, c, _ = read_times_from_nccl_log(logfile, end=512*1024*1024, original=True, bw=True)
                    #_, c, _ = read_times_from_nccl_log(logfile, end=512*1024*1024, original=True)
                    c = c[0]
                except:
                    c = 0.0
                    error += 1
                tmp_cs.append(c)
            if error > 1:
                raise
            print('task[msg_size=%d, #jobs=%d, #node=%d, error: %d' % \
                (msg_size, n_job, nnode, error))
            #jc = np.max(tmp_cs) # us
            c = np.max(tmp_cs) # us
            throughput = np.sum(tmp_cs) #msg_size*(n_job-error)/1e6 / c
            #throughput = msg_size*(n_job-error)*8*2/1e9 / c
            job_data[i].append(throughput)
        label = 'All-Reduce (%d task)'% (n_job)
        if n_job > 1:
            label = 'All-Reduce (%d tasks)'% (n_job)
        ax.plot(msg_sizes, job_data[i], label=label)
    iperfline = [iperfbw] * len(job_data[0])
    ax.plot(msg_sizes, iperfline, label='Link Bandwidth (%.2f Gbps)' % iperfbw, linestyle='--')
    ax.legend(prop={'size': FONTSIZE-2})
    #ax.legend()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_xlabel('Message size [bytes]')
    ax.set_ylabel('Bandwidth [Gbps]')
    ax.set_ylim(top=10)
    utils.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'allreduce-results-%d'%nnode), bbox_inches='tight')
    plt.show()

def plot_time_penalty():
    #fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(6.6,5.0))
    iperfbw=9.43
    msg_sizes= list(range(512, 4194304, 32768))
    job_list = [1, 2]
    job_data = [[] for j in job_list]
    nnode=8
    rdma=0
    #fig, ax = plt.subplots(figsize = (12, 8))
    for i, n_job in enumerate(job_list):
        for msg_size in msg_sizes:
            folder='logs/infocom21-v2/nccl_rdma%d_job_nw%d_n%d_s%d' % (rdma, nnode, n_job, msg_size)
            tmp_cs = []
            error = 0
            for k in range(1, n_job+1):
                fn = 'nccl_job_%d.log' % k
                logfile = os.path.join(folder, fn)
                try:
                    #_, c, _ = read_times_from_nccl_log(logfile, end=512*1024*1024, original=True, bw=True)
                    _, c, _ = read_times_from_nccl_log(logfile, end=512*1024*1024, original=True)
                    c = c[0]
                except:
                    c = 0.0
                    error += 1
                tmp_cs.append(c)
            if error > 1:
                raise
            print('task[msg_size=%d, #jobs=%d, #node=%d, error: %d' % \
                (msg_size, n_job, nnode, error))
            #jc = np.max(tmp_cs) # us
            c = np.mean(tmp_cs) # us
            #throughput = np.sum(tmp_cs) #msg_size*(n_job-error)/1e6 / c
            #throughput = msg_size*(n_job-error)*8*2/1e9 / c
            job_data[i].append(c)
        label = 'Measured All-Reduce (%d task)'% (n_job)
        if n_job > 1:
            label = 'Measured All-Reduce (%d tasks)'% (n_job)
        ax.plot(msg_sizes, job_data[i], label=label)
    ax.plot(msg_sizes, np.array(job_data[1])-np.array(job_data[0]), label=r'Measured $t_{penalty}$')
    a, b = _fit_linear_function(msg_sizes, job_data[0])
    iperfline = [iperfbw] * len(job_data[0])
    #a = 0.001368709228921283
    #b = 1.6569786551462448e-09
    predicted_one = [a + s * b for s in msg_sizes]
    predicted_two = [a + 1.5 *s * b for s in msg_sizes]
    time_penalty = np.array(predicted_two) - np.array(predicted_one)
    ax.plot(msg_sizes, predicted_one, label='Predicted All-Reduce (1 task)', linestyle='--')
    ax.plot(msg_sizes, predicted_two, label='Predicted All-Reduce (2 tasks)', linestyle='--')
    ax.plot(msg_sizes, time_penalty, label=r'Predicted $t_{penalty}$', linestyle='--')
    ax.legend()
    #ax.legend()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_xlabel('Message size [bytes]')
    ax.set_ylabel('Time [s]')
    #utils.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'contention-results-%d'%nnode), bbox_inches='tight')
    plt.show()


def _fit_linear_function(x, y):
    X = np.array(x).reshape((-1, 1))
    Y = np.array(y)
    print('x: ', X)
    print('y: ', Y)
    model = LinearRegression()
    model.fit(X, Y)
    alpha = model.intercept_
    beta = model.coef_[0]
    #A = np.vstack([X, np.ones(len(X))]).T
    #beta, alpha = np.linalg.lstsq(A, Y, rcond=None)[0]
    return alpha, beta

def plot_p2platency(nccl=True):
    #fig, ax = plt.subplots(figsize=(5,3.8))
    fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=(5,4.2))
    rdma=0
    nworkers=8
    filename = 'logs/infocom21/nccl-rdma%d-nworkers%d-v2.log' % (rdma, nworkers)
    sizes, comms, errors = read_times_from_nccl_log(filename, start=8*1024, end=4*1024*1024, original=True)
    alpha, beta = _fit_linear_function(sizes, comms)
    print('alpha: ', alpha, ', beta: ', beta)

    #ax.errorbar(sizes, comms, errors, label='Measured', linewidth=0.5)
    #sizes = sizes[comms<0.02]
    #comms = comms[comms<0.02]
    ax.plot(sizes, comms, label='Measured', linewidth=1.5)
    ax.plot(sizes, alpha+np.array(sizes)*beta, label=r'Predicted ($a$=%.1e, $b$=%.1e)'%(Decimal(alpha), Decimal(beta)), linewidth=1.5)
    #ax.grid(linestyle=':')
        
    plt.xlabel('Size of parameters [bytes]')
    plt.ylabel(r'All-Reduce time [s]')
    plt.ylim(bottom=0, top=plt.ylim()[1]*1.2)
    plt.xlim(left=0)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.legend(ncol=1, loc=2, prop={'size': FONTSIZE-2})
    utils.update_fontsize(ax, fontsize=FONTSIZE)
    plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'allreducecommtime-%d'%nworkers), bbox_inches='tight')
    plt.show()

def plot_bandwidth_allreduce():
    fig, ax = plt.subplots()
    iperfbw=9.43
    rdma=0
    nworkers=8
    filename = 'logs/infocom21/nccl-rdma%d-nworkers%d-v3.log' % (rdma, nworkers)
    sizes, bandwidths, errors = read_times_from_nccl_log(filename, start=16*1024, end=4*1024*1024, original=True, bw=True)
    iperfline = [iperfbw] * len(bandwidths)
    ax.plot(sizes, bandwidths, label='All-Reduce (1 task)')
    ax.plot(sizes, iperfline, label='Link Bandwidth (%.2f Gbps)' % iperfbw, linewidth=1)
    ax.set_ylabel('Bandwidth [Gbps]')
    ax.set_xlabel(r'Size of messages [bytes]')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    plt.legend(ncol=1, loc=4, prop={'size': FONTSIZE-2})
    utils.update_fontsize(ax, fontsize=FONTSIZE)
    plt.savefig('%s/%s.pdf' % (OUTPUT_PATH, 'allreduce-results-%d'%nworkers), bbox_inches='tight')
    plt.show()

def plot_multi_breakdown():
    fig, ax = plt.subplots(figsize=(7.6,4.4))
    xticklabels = ['ResNet-152', 'DenseNet-201', 'Inception-v4', 'BERT-Base']
    dnns = ['resnet152', 'densenet201', 'inceptionv4', 'bertbase']
    algos = ['wfbp', 'mg-wfbp', 'asc-wfbp']
    data = {'resnet152':  # [compute, communication]
                    {
                     'mg-wfbp': [0.273738238, 0.436140445],
                     'asc-wfbp': [0.273738238, 0.250556473],
                     'wfbp': [0.273738238, 0.682199029],
                     }, 
            'densenet201':  # [compute, communication]
                    {
                     'mg-wfbp': [0.218878249, 0.156268293],
                     'asc-wfbp': [0.218878249, 0.094818409],
                     'wfbp':    [0.218878249, 0.559296384],
                     }, 
            'inceptionv4':  # [compute, communication]
                    {
                     'mg-wfbp': [0.18946122, 0.320422762],
                     'asc-wfbp': [0.18946122, 0.187633238],
                     'wfbp':    [0.18946122, 0.529135272],
                     }, 
            'bertbase':  # [compute, communication]
                    {
                     'mg-wfbp': [0.30548926, 0.756861606],
                     'asc-wfbp': [0.30548926, 0.618740173],
                     'wfbp': [0.30548926, 0.704721365],
                     }, 
            }
    def Smax(times):
        tf = times[0]; tb=times[1]; tc=times[2]
        r = tc/tb
        s = 1+1.0/(tf/min(tc,tb)+max(r,1./r))
        return s
    count = len(dnns)
    width = 0.27; margin = 0.02
    s = (1 - (width*count+(count-1) *margin))/2+width
    ind = np.array([s+i+1 for i in range(count)])
    labels=['WFBP', 'MG-W.', 'ASC-W.']
    
    for i, algo in enumerate(algos):
        newind = ind+s*width+(s+1)*margin
        forward = []; backward=[];sparse=[];commu=[]
        for dnn in dnns:
            d = data[dnn]
            ald = d[algo]
            if algo.find('topk') >= 0:
                print('dnn: ', dnn, ' s: ', Smax(ald))
            forward.append(ald[0])
            commu.append(ald[1])
        p1 = ax.bar(newind, forward, width, color=Color.backward_color,edgecolor='black', label='Computation')
        p2 = ax.bar(newind, commu, width, bottom=np.array(forward), color=Color.comm_color,edgecolor='black', label='Communication')
        #p3 = ax.bar(newind, sparse, width, bottom=np.array(forward)+np.array(backward), color=Color.compression_color,edgecolor='black', label='Sparsification')
        #p4 = ax.bar(newind, commu, width, bottom=np.array(forward)+np.array(backward)+np.array(sparse), color=Color.comm_color,edgecolor='black', label='Communication')
        s += 1 
        #ax.text(4, 4, 'ehhlo', color='b')
        utils.autolabel(p2, ax, labels[i], 0, 8)
    ax.set_ylim(top=ax.get_ylim()[1]*1.05)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend((p1[0], p2[0]), (labels[0],labels[1]), ncol=2, handletextpad=0.2, columnspacing =1., loc='upper center')
    ax.set_ylabel('Time [s]')
    ax.set_xticks(newind-width-margin/2)
    ax.set_xticklabels(xticklabels)
    plt.savefig('%s/timebreakdown.pdf' % (OUTPUT_PATH), bbox_inches='tight')
    plt.show()


def plot_single_breakdown():
    names = ['Computation', 'Communication']
    resnet152=[0.273738238,    1.14082178429241]
    densenet201=[0.218878249,    0.51504040359721]
    bertbase=[0.30548926,    0.763962436984739]
    def _get_ratio(data):
        return [v/np.sum(data) for v in data]
    ratio1= _get_ratio(resnet152) 
    ratio2= _get_ratio(densenet201) 
    ratio3= _get_ratio(bertbase) 
    datas = [ratio1, ratio2, ratio3]
    for d in datas:
        print('ratios: ', d)
    communications = [ratio1[1], ratio2[1], ratio3[1]] 
    computes = [ratio1[0], ratio2[0], ratio3[0]] 
    computes = np.array(computes)
    communications= np.array(communications)
    fig, ax = plt.subplots(figsize=(4.8,3.4))

    count = len(datas)
    ind = np.arange(count)
    width = 0.35
    margin = 0.05
    xticklabels = ['ResNet-152', 'DenseNet-201', 'BERT-Base']
    #ind = np.array([s+i+1 for i in range(count)])
    newind = np.arange(count)
    p1 = ax.bar(newind, computes, width, color=Color.comp_color,hatch='x', label=names[0])
    p2 = ax.bar(newind, communications, width, bottom=computes, color=Color.opt_comm_color,hatch='-', label=names[1])

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
    utils.update_fontsize(ax, 10)
    ax.legend((p1[0], p2[0]), tuple(names), ncol=9, bbox_to_anchor=(1, -0.1))#, handletextpad=0.2, columnspacing =1.)
    #ax.legend((p1[0], p2[0]), (labels[0],labels[1] ), ncol=2, handletextpad=0.2, columnspacing =1.)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.19, top=0.94)
    #plt.savefig('%s/breakdown.pdf' % (OUTPUT_PATH))
    plt.show()



if __name__ == '__main__':
    #plot_multiple_jobs()
    #plot_allreduce_throughput()
    #plot_time_penalty()
    #plot_p2platency()
    #plot_bandwidth_allreduce()
    plot_multi_breakdown()

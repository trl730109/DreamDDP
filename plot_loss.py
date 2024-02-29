# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import datetime
import itertools
import utils as u
#markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)


#OUTPUTPATH='/media/sf_Shared_Data/tmp/p2psgd'
OUTPUTPATH='/media/sf_Shared_Data/tmp/nmi'

EPOCH = True
FONTSIZE=12

fig, ax = plt.subplots(1,1,figsize=(5,3.8))
ax2 = None

STANDARD_TITLES = {
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG-16',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50',
        'lstmptb': 'LSTM-PTB',
        'lstm': 'LSTM-PTB',
        'lstman4': 'LSTM-AN4'
        }

fixed_colors = {
        #'S-SGD': 'r',
        #'gTopK': 'g'
        'S-SGD': '#ff3300',
        'gtopk': 'g',
        'gtop-k': 'g',
        'topk': 'r',
        'top-k': 'r',
        'dense': 'C3',
        'blue': 'b',
        0.001: 'C2',
        0.002: 'C5',
        0.00025: 'C3',
        0.0001: 'C0',
        0.00005: 'C1',
        0.00001: 'C4',
        }

STANDARD_MARKERS = {'dense':'o',
        'topk':'x',
        'top-k':'x',
        'gtopk':'^',
        'gtop-k':'^',
        }

def get_real_title(title):
    return STANDARD_TITLES.get(title, title)

def seconds_between_datetimestring(a, b):
    a = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    b = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    delta = b - a 
    return delta.days*86400+delta.seconds
sbd = seconds_between_datetimestring

def get_loss(line, isacc=False):
    if EPOCH:
        #if line.find('Epoch') > 0 and line.find('acc:') > 0:
        valid = line.find('val acc: ') > 0 if isacc else line.find('loss: ') > 0
        #if line.find('Epoch') > 0 and line.find('loss:') > 0 and not line.find('acc:')> 0:
        if line.find('Epoch') > 0 and valid: 
            items = line.split(' ')
            loss = float(items[-1])
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            return loss, t
    else:
        if line.find('average forward') > 0:
            items = line.split('loss:')[1]
            loss = float(items[1].split(',')[0])
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            return loss, t
    return None, None

def read_losses_from_log(logfile, isacc=False):
    f = open(logfile)
    losses = []
    times = []
    average_delays = []
    lrs = []
    i = 0
    time0 = None 
    max_epochs = 200
    counter = 0
    for line in f.readlines():
        #if line.find('average forward') > 0:
        valid = line.find('val acc: ') > 0 if isacc else line.find('average loss: ') > 0
        if line.find('Epoch') > 0 and valid:
        #if not time0 and line.find('INFO [  100]') > 0:
            t = line.split(' I')[0].split(',')[0]
            t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            if not time0:
                time0 = t
        if line.find('lr: ') > 0:
            try:
                lr = float(line.split(',')[-2].split('lr: ')[-1])
                lrs.append(lr)
            except:
                pass
        if line.find('average delay: ') > 0:
            delay = int(line.split(':')[-1])
            average_delays.append(delay)
        loss, t = get_loss(line, isacc)
        if loss and t:
            counter += 1
            losses.append(loss)
            times.append(t)
        if counter > max_epochs:
            break

        #if line.find('Epoch') > 0 and line.find('acc:') > 0:
        #    items = line.split(' ')
        #    loss = float(items[-1])
        #    #items = line.split('loss:')[1]
        #    #loss = float(items[1].split(',')[0])

        #    losses.append(loss)
        #    t = line.split(' I')[0].split(',')[0]
        #    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        #    times.append(t)
    f.close()
    if not EPOCH:
        average_interval = 10
        times = [times[t*average_interval] for t in range(len(times)/average_interval)]
        losses = [np.mean(losses[t*average_interval:(t+1)*average_interval]) for t in range(len(losses)/average_interval)]
    if len(times) > 0:
        t0 = time0 if time0 else times[0] #times[0]
        for i in range(0, len(times)):
            delta = times[i]- t0
            times[i] = delta.days*86400+delta.seconds
    return losses, times, average_delays, lrs

def read_norm_from_log(logfile):
    f = open(logfile)
    means = []
    stds = []
    for line in f.readlines():
        if line.find('gtopk-dense norm mean') > 0:
            items = line.split(',')
            mean = float(items[-2].split(':')[-1])
            std = float(items[--1].split(':')[-1])
            means.append(mean)
            stds.append(std)
    print('means: ', means)
    print('stds: ', stds)
    return means, stds

def plot_loss(logfile, label, isacc=False, title='ResNet-20'):
    losses, times, average_delays, lrs = read_losses_from_log(logfile, isacc=isacc)
    norm_means, norm_stds = read_norm_from_log(logfile)

    #print('times: ', times)
    #print('Learning rates: ', lrs)
    if len(average_delays) > 0:
        delay = int(np.mean(average_delays))
    else:
        delay = 0
    if delay > 0:
        label = label + ' (delay=%d)' % delay
    #plt.plot(losses, label=label, marker='o')
    #plt.xlabel('Epoch')
    #plt.title('ResNet-20 loss')
    if isacc:
        ax.set_ylabel('Top-1 Validation Accuracy')
    else:
        ax.set_ylabel('Training loss')
    #plt.title('ResNet-50')
    ax.set_title(get_real_title(title))
    marker = markeriter.next()
    color = coloriter.next()
    #print('marker: ', marker)
    #ax.plot(losses[0:180], label=label, marker=marker, markerfacecolor='none')
    ax.plot(losses, label=label, marker=marker, markerfacecolor='none', color=color)
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)
    ax.set_xlabel('Epoch')
    #plt.plot(times, losses, label=label, marker=markeriter.next())
    #plt.xlabel('Time [s]')
    ax.grid(linestyle=':')
    if len(lrs) > 0:
        lr_indexes = [0]
        lr = lrs[0]
        for i in range(len(lrs)):
            clr = lrs[i]
            if lr != clr:
                lr_indexes.append(i)
                lr = clr
        #for i in lr_indexes:
        #    if i < len(losses):
        #        ls = losses[i]
        #        ax.text(i, ls, 'lr=%f'%lrs[i])
    u.update_fontsize(ax, FONTSIZE)

def plot_loss_with_host(hostn, nworkers, hostprefix, baseline=False):
    if not baseline or nworkers == 64:
        port = 5922
    else:
        port = 5945
    for i in range(hostn, hostn+1):
        for j in range(2, 3):
            host='%s%d-%d'%(hostprefix, i, port+j)
            if baseline:
                logfile = './ad-sgd-%dn-%dw-logs/'%(nworkers/4, nworkers)+host+'.log'
            else:
                logfile = './%dnodeslogs/'%nworkers+host+'.log'
                if nworkers == 256 and hostn < 48:
                    host='%s%d.comp.hkbu.edu.hk-%d'%(hostprefix, i, port+j)
                    logfile = './%dnodeslogs/'%nworkers+host+'.log'
                #csr42.comp.hkbu.edu.hk-5922.log
                #logfile = './%dnodeslogs-w/'%nworkers+host+'.log'
            label = host+' ('+str(nworkers)+' workers)'
            if baseline:
                label += ' Baseline'
            plot_loss(logfile, label) 

def plot_with_params(dnn, nworkers, bs, lr, hostname, legend, isacc=False, prefix='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False):
    postfix='5922'
    if prefix.find('allreduce')>=0:
        postfix='0'
    if sparsity:
        logfile = './logs/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (prefix, dnn, nworkers, bs, lr, sparsity)
    elif nsupdate:
        logfile = './logs/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (prefix, dnn, nworkers, bs, lr, nsupdate)
    else:
        logfile = './logs/%s/%s-n%d-bs%d-lr%.4f' % (prefix, dnn, nworkers, bs, lr)
    if sg is not None:
        logfile += '-sg%.2f' % sg
    if density is not None:
        logfile += '-ds%s' % str(density)
    logfile += '/%s-%s.log' % (hostname, postfix)
    print('logfile: ', logfile)
    if force_legend:
        l = legend
    else:
        l = legend+ '(lr=%.4f, bs=%d, %d workers)'%(lr, bs, nworkers)
    plot_loss(logfile, l, isacc=isacc, title=dnn) 

def resnet20():
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'Allreduce', prefix='allreduce')
    #plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Allreduce', prefix='allreduce-baseline-wait-dc1-model-debug', nsupdate=1)
    plot_with_params('resnet20', 4, 32, 0.1, 'hpclgpu', '(Ref 1/4 data)', prefix='compression-modele',sparsity=0.01)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Spar', prefix='compression-dc1-model-debug',sparsity=0.999)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Gradients', prefix='adpsgd-dc1-grad-debug',sparsity=None)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu13', 'ADPSGD+Gradients Sync', prefix='adpsgd-wait-dc1-grad-debug',sparsity=None)
    #plot_with_params('resnet20', 4, 32, 0.1, 'gpu21', 'Allreduce', prefix='allreduce')
    plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Sequence d=0.1, lr=0.01', prefix='allreduce-comp-sequence-baseline-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    plot_with_params('resnet20', 4, 32, 0.1, 'gpu11', 'Sequence d=0.1, lr=0.1, wp', prefix='allreduce-comp-sequence-baseline-gwarmup-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    plot_with_params('resnet20', 4, 32, 0.01, 'gpu17', 'Sequence d=0.1, lr=0.01, wp', prefix='allreduce-comp-sequence-baseline-gwarmup-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.1, force_legend=True)
    #plot_with_params('resnet20', 4, 32, 0.01, 'gpu11', 'Sequence density=0.1', prefix='allreduce-comp-sequence-baseline-wait-dc1-model-debug', nsupdate=1, sg=1.5, density=0.01, force_legend=True)

def vgg16():
    #plot_with_params('vgg16', 4, 32, 0.1, 'gpu17', 'Allreduce', prefix='allreduce')
    #plot_with_params('vgg16', 4, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 128, 1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', title='VGG16', sparsity=0.95)
    #plot_with_params('vgg16', 4, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='compression-modele', title='VGG16', sparsity=0.98)
    plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.01, 'hpclgpu', 'ADPSGD+DC4', prefix='baseline-dc4-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-wait-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 8, 32, 0.1, 'MGD', 'ADPSGD ', prefix='baseline-modelmgd', title='VGG16')
    plot_with_params('vgg16', 16, 32, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.1, 'gpu20', 'ADPSGD ', prefix='baseline-modelk80', title='VGG16')
    #plot_with_params('vgg16', 16, 32, 0.0005, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl', title='VGG16')
    plot_with_params('vgg16', 4, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    plot_with_params('vgg16', 8, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')
    plot_with_params('vgg16', 16, 32, 0.1, 'gpu14', 'ASGD', prefix='baseline-wait-ps-dc1-modelk80')

def mnistnet():
    plot_with_params('mnistnet', 1, 512, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')
    plot_with_params('mnistnet', 1, 512, 0.1, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')
    plot_with_params('mnistnet', 1, 64, 0.01, 'hpclgpu', 'ADPSGD ', prefix='baseline-modelhpcl')

def plot_one_worker():
    def _plot_with_params(bs, lr, isacc=True):
        logfile = './logs/resnet20/accresnet20-bs%d-lr%s.log' % (bs, str(lr))
        t = '(lr=%.4f, bs=%d)'%(lr, bs)
        plot_loss(logfile, t, isacc=isacc, title='resnet20') 
    _plot_with_params(32, 0.1)
    _plot_with_params(32, 0.01)
    _plot_with_params(32, 0.001)
    _plot_with_params(64, 0.1)
    _plot_with_params(64, 0.01)
    _plot_with_params(64, 0.001)
    _plot_with_params(128, 0.1)
    _plot_with_params(128, 0.01)
    _plot_with_params(128, 0.001)
    _plot_with_params(256, 0.1)
    _plot_with_params(256, 0.01)
    _plot_with_params(256, 0.001)
    _plot_with_params(512, 0.1)
    _plot_with_params(512, 0.01)
    _plot_with_params(512, 0.001)
    _plot_with_params(1024, 0.1)
    _plot_with_params(1024, 0.01)
    _plot_with_params(1024, 0.001)
    _plot_with_params(2048, 0.1)

def resnet50():
    plot_loss('baselinelogs/accresnet50-lr0.01-c40,70.log', 'allreduce 4 GPUs', isacc=False, title='ResNet-50') 

    plot_with_params('resnet50', 8, 64, 0.01, 'gpu10', 'allreduce 8 GPUs', prefix='allreduce-debug')
    plot_with_params('resnet50', 8, 64, 0.01, 'gpu16', 'ADPSGD', prefix='baseline-dc1-modelk80')

def plot_norm_diff():
    network = 'resnet20'
    bs =32
    #network = 'vgg16'
    #bs = 128
    path = './logs/allreduce-comp-gtopk-baseline-gwarmup-dc1-model-normtest/%s-n4-bs%d-lr0.1000-ns1-sg1.50-ds0.001' % (network,bs)
    epochs = 80
    arr = None
    arr2 = None
    arr3 = None
    for i in range(1, epochs):
        fn = '%s/gtopknorm-rank0-epoch%d.npy' % (path, i)
        fn2 = '%s/randknorm-rank0-epoch%d.npy' % (path, i)
        fn3 = '%s/upbound-rank0-epoch%d.npy' % (path, i)
        fn4 = '%s/densestd-rank0-epoch%d.npy' % (path, i)
        if arr is None:
            arr = np.load(fn)
            arr2 = np.load(fn2)
            arr3 = np.load(fn3)
            arr4 = np.load(fn4)
        else:
            arr = np.concatenate((arr, np.load(fn)))
            arr2 = np.concatenate((arr2, np.load(fn2)))
            arr3 = np.concatenate((arr3, np.load(fn3)))
            arr4 = np.concatenate((arr4, np.load(fn4)))
    #plt.plot(arr-arr2, label='||x-gtopK(x)||-||x-randomK(x)||')
    plt.plot(arr4, label='Gradients std')
    plt.xlabel('# of iteration')
    #plt.ylabel('||x-gtopK(x)||-||x-randomK(x)||')
    plt.title(network)
    #plt.plot(arr2, label='||x-randomK(x)||')
    #plt.plot(arr3, label='(1-K/n)||x||')

def loss(network):
    # Convergence
    gtopk_name = 'gTopKAllReduce'
    dense_name = 'DenseAllReduce'

    # resnet20
    if network == 'resnet20':
        plot_with_params(network, 4, 32, 0.1, 'gpu21', dense_name, prefix='allreduce', force_legend=True)
        plot_with_params(network, 4, 32, 0.1, 'gpu13', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'vgg16':
        # vgg16
        plot_with_params(network, 4, 128, 0.1, 'gpu13', dense_name, prefix='allreduce-baseline-dc1-model-icdcs', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 128, 0.1, 'gpu10', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'alexnet':
        plot_with_params(network, 8, 256, 0.01, 'gpu20', dense_name, prefix='allreduce-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, force_legend=True)
        plot_with_params(network, 8, 256, 0.01, 'gpu20', gtopk_name, prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=0.001, force_legend=True)
    elif network == 'resnet50':
        plot_with_params(network, 8, 64, 0.01, 'gpu10', dense_name, prefix='allreduce-debug', force_legend=True)
    #todo zhtang an4 ==============
    elif network == 'lstman4':
        plot_with_params(network, 8, 8, 1, 'gpu10', dense_name, prefix='allreduce-debug', force_legend=True)

def communication_speed():
    pass

def icdcs2019all():
    def convergence():
        #network = 'resnet20'
        #network = 'vgg16'
        #network = 'alexnet'
        #network = 'resnet50'
        network = 'lstman4'
        icdcs2019(network=network)

        ax.set_xlim(xmin=-1)
        ax.legend(fontsize=FONTSIZE)
        plt.subplots_adjust(bottom=0.18, left=0.2, right=0.96, top=0.9)
        #plt.savefig('%s/%s_convergence.pdf' % (OUTPUTPATH, network))
        plt.show()

    def communication_speed():
        pass

    convergence()

def read_speed(logfile):
    f = open(logfile, 'r')
    speeds = []
    for line in f.readlines():
        if line.find('Speed') > 0:
            speedstr = line.split('Speed: ')[-1].split(' ')[0]
            speed = float(speedstr)
            speeds.append(speed)
    avg_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    f.close()
    print('avg speed: ', avg_speed, ' std: ', std_speed)
    return avg_speed

def _get_log_filename(prefix, dnn, nw, bs, lr, sparsity, nsupdate, sg, density, hostname):
     postfix='5922'
     if prefix.find('allreduce')>=0:
         postfix='0'
     if sparsity:
         logfile = './logs/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (prefix, dnn, nw, bs, lr, sparsity)
     elif nsupdate:
         logfile = './logs/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (prefix, dnn, nw, bs, lr, nsupdate)
     else:
         logfile = './logs/%s/%s-n%d-bs%d-lr%.4f' % (prefix, dnn, nw, bs, lr)
     if sg is not None:
         logfile += '-sg%.2f' % sg
     if density is not None:
         logfile += '-ds%s' % str(density)
     if nw > 1:
         logfile += '/%s-%s.log' % (hostname, postfix)
     else:
         logfile += '/%s.log' % (hostname)
     print('logfile: ', logfile)
     return logfile

def plotspeed_with_params(dnn, nworkers, bs, lr, hostname, legend, isacc=False, prefix='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False):
    nws = []
    avg_speeds = []
    if type(nworkers) is list:
        for nw in nworkers:
            if nw == 1:
                if dnn in ['vgg16', 'resnet20']:
                    logfile = _get_log_filename('singlegpu-baseline-gwarmup-dc1-model-ijcai-wu2norm', dnn, nw, bs, lr, None, 1, None, None, hostname)
                else:
                    logfile = _get_log_filename('singlegpu-baseline-gwarmup-dc1-model-general-1n2w', dnn, nw, bs, lr, None, 1, None, None, hostname)
            else:
                logfile = _get_log_filename(prefix, dnn, nw, bs, lr, sparsity, nsupdate, sg, density, hostname)
            avg_speed = read_speed(logfile)
            avg_speeds.append(avg_speed*nw)
            #avg_speeds.append(avg_speed)
            nws.append(nw)
    else:
        logfile = _get_log_filename(prefix, dnn, nworkers, bs, lr, sparsity, nsupdate)
        avg_speed = read_speed(logfile)
        avg_speeds.append(avg_speed)
        nws.append(nworkers)
    algo = 'top-k'
    if logfile.find('gtopk') > 0:
        algo = 'gtop-k'
    elif logfile.find('-topk') > 0:
        algo = 'top-k'
    else:
        algo = 'dense'
    print('algo: ', algo, ', avgs: ', avg_speeds)
    eff = [avg/avg_speeds[0] for avg in avg_speeds]
    start = 0
    #ax.plot(nws[start:], eff[start:], label=legend, color=fixed_colors[algo])
    opts = [avg_speeds[0]*2**i for i in range(len(avg_speeds))]
    ax.plot(nws, avg_speeds, label=legend, color=fixed_colors[algo], marker=STANDARD_MARKERS[algo], linewidth=1)
    #ax.plot(nws, opts, label='Ideal', color='black')
    ylabel = 'throughput [samples/s]'
    #if dnn == 'lstm':
    #    ylabel = 'throughput [sentences/s]'
    #elif dnn == 'lstman4':
    #    ylabel = 'throughput [audio/s]'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('# of workers')
    ax.set_xticks(nws[start:])
    ax.set_title('%s'%STANDARD_TITLES.get(dnn, dnn))
    u.update_fontsize(ax, fontsize=FONTSIZE)
    #plt.subplots_adjust(bottom=0.15, left=0.17, right=0.90, top=0.88)
    plt.subplots_adjust(bottom=0.15, left=0.19, right=0.98, top=0.88)
    return np.array(avg_speeds)


def plot_comm_overheads():
    def predict_comm_time(p, size, density, algo):
        alpha = 0.436*0.001
        beta = 9.e-9
        k = density *size
        if algo == 'gtopk':
            return np.log2(p)*2*alpha + 4*k*np.log2(p)*beta
        else:
            return np.log2(p)*2*alpha + 2*k*(p-1)*beta
    def _read_comm_size(logfile):
        print('logfile: ', logfile)
        size = 0 
        t = 0 
        with open(logfile, 'r') as f:
            for line in f.readlines():
                if line.find('allreducer.py:547') > 0:
                    size = int(line.split('...')[-1].split(']')[0].split('[')[-1]) * 4
                    t = float(line.split('...')[-1].split(',')[3])
                    print('size: ', size)
                    break
        return size, t

    dnn='resnet20';density=0.001;bs=128;lr=0.1;nworkers=[8, 16, 32]
    #dnn='vgg16';density=0.001;bs=128;lr=0.1;nworkers=[8, 16, 32]

    real_gtopkcomm_times = []
    real_topkcomm_times = []
    gtopk_times = []
    topk_times = []
    for nw in nworkers:
        prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-nmifinal'
        logfile = _get_log_filename(prefix, dnn, nw, bs, lr, None, 1, 1.5, density, 'MGD')
        size, gtopkcomm_time = _read_comm_size(logfile)

        prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nmifinal'
        logfile = _get_log_filename(prefix, dnn, nw, bs, lr, None, 1, 1.5, density, 'MGD')
        size, topkcomm_time = _read_comm_size(logfile)

        real_gtopkcomm_times.append(gtopkcomm_time)
        real_topkcomm_times.append(topkcomm_time)
        gtopk_times.append(predict_comm_time(nw, size, density, 'gtopk'))
        topk_times.append(predict_comm_time(nw, size, density, 'topk'))
    ax.plot(nworkers, real_gtopkcomm_times, label='real gTop-k', color='r')
    ax.plot(nworkers, real_topkcomm_times, label='real Top-k', color='C4')
    ax.plot(nworkers, gtopk_times, label='gTop-k', color='g')
    ax.plot(nworkers, topk_times, label='Top-k', color='b')
    ax.set_title('Size: %.3f KB' % (size*density*1e-3))
    print('gtopk real: ', real_gtopkcomm_times)
    print('gtopk: ', gtopk_times)
    print('topk: ', topk_times)


def plot_throughputs():
    exp='nmi2'
    #exp='nmifinal'
    dnn='resnet20';density=0.001;bs=128;lr=0.1;nworkers=[4, 8, 16, 32, 64]
    #dnn='resnet110';density=0.001;bs=128;lr=0.1;nworkers=[4, 8, 16, 32, 64]
    #dnn='vgg16';density=0.001;bs=128;lr=0.1;nworkers=[4, 8, 16, 32, 64]
    dnn='lstm';density=0.001;bs=100;lr=1.0;nworkers=[4, 8, 16, 32, 64]
    dnn='lstman4';density=0.001;bs=4;lr=0.0003;nworkers=[4, 8, 16, 32, 64]
    gtopk_speeds = plotspeed_with_params(dnn=dnn, nworkers=nworkers, bs=bs, lr=lr, hostname='MGD', legend=r'gTop-$k$ S-SGD', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-'+exp, title=dnn, sparsity=None, nsupdate=1, sg=1.5, density=density, force_legend=True)
    topk_speeds = plotspeed_with_params(dnn=dnn, nworkers=nworkers, bs=bs, lr=lr, hostname='MGD', legend=r'Top-$k$ S-SGD', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-'+exp, title=dnn, sparsity=None, nsupdate=1, sg=1.5, density=density, force_legend=True)
    #dense_speeds = plotspeed_with_params(dnn=dnn, nworkers=nworkers, bs=bs, lr=lr, hostname='MGD', legend=r'Dense', prefix='allreduce-baseline-gwarmup-dc1-model-nmifinal', title=dnn, sparsity=None, nsupdate=1, sg=None, density=None, force_legend=True)
    #topk_speedups = gtopk_speeds/topk_speeds
    #dense_speedups = gtopk_speeds/dense_speeds
    #print(dnn, ', dense: ', dense_speeds[-1])
    #print(dnn, ', topk: ', topk_speeds[-1])
    #print(dnn, ', gtopk: ', gtopk_speeds[-1])
    #print(dnn, ', topk_speedups: ', topk_speedups[-1])
    #print(dnn, ', dense_speedups: ', dense_speedups[-1])
    ax.grid(linestyle=':')
    plt.legend(fontsize=FONTSIZE)
    plt.savefig('%s/%s_throughput.pdf' % (OUTPUTPATH, dnn))


if __name__ == '__main__':
    #resnet20()
    #plot_norm_diff()
#VGG and ResNet
    #plot_comm_overheads()
    plot_throughputs()
    plt.show()

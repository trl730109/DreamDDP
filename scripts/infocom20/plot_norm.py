# -*- coding: utf-8 -*-
from __future__ import print_function
import time
from matplotlib import rcParams
FONT_FAMILY='DejaVu Serif'
rcParams["font.family"] = FONT_FAMILY 
from mpl_toolkits.axes_grid.inset_locator import inset_axes, zoomed_inset_axes
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
#rc('text', usetex=True)
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
fixed_colors = {
        #'S-SGD': 'r',
        #'gTopK': 'g'
        'S-SGD': '#ff3300',
        'ssgd': '#ff3300',
        'gTopK': '#009900',
        #'blue': '#355bff',
        'blue': 'b',
        0.001: 'C2',
        0.002: 'C5',
        0.00025: 'C3',
        0.0001: 'C0',
        0.00005: 'C1',
        0.00001: 'C4',
        }

#OUTPUTPATH='/media/sf_Shared_Data/tmp/p2psgd'
#OUTPUTPATH='/media/sf_Shared_Data/tmp/icdcs2019'
OUTPUTPATH='/media/sf_Shared_Data/tmp/ijcai2019cr'
LOGHOME='/media/sf_Shared_Data/gpuhome/repositories/p2p-dl/logs'
#LOGHOME='/home/shshi/host143/repos/p2p-dl/logs'
#LOGHOME='/home/shshi/host144/repos/p2p-dl/logs'

EPOCH = True
FONTSIZE=16
num_batches_per_epoch = None
global_max_epochs=150
global_density=0.001
NFIGURES=4;NFPERROW=2
#NFIGURES=6;NFPERROW=2
#NFIGURES=1;NFPERROW=1
USE_BATCH_X = False
#USE_BATCH_X = True
#FIGSIZE=(5*NFPERROW,3.8*NFIGURES/NFPERROW)
PLOT_NORM=False
PLOT_NORM=True
if PLOT_NORM:
    #FIGSIZE=(5*NFPERROW,3.1*NFIGURES/NFPERROW)
    FIGSIZE=(5*NFPERROW,3.2*NFIGURES/NFPERROW)
else:
    #FIGSIZE=(5*NFPERROW,2.9*NFIGURES/NFPERROW)
    FIGSIZE=(5*NFPERROW,3.0*NFIGURES/NFPERROW)

fig, group_axs = plt.subplots(NFIGURES/NFPERROW, NFPERROW,figsize=FIGSIZE)
if NFIGURES > 1 and PLOT_NORM:
    ax = None
    group_axtwins = []
    for i in range(NFIGURES/NFPERROW):
        tmp = []
        for a in group_axs[i]:
            tmp.append(a.twinx())
        group_axtwins.append(tmp)
    global_index = 0
else:
#ax1 = ax
    ax = group_axs
    ax1 = ax
    #ax1 = group_axs.twinx();tmp = ax1;ax1 = group_axs; ax=tmp
    global_index = None
#fig, ax = plt.subplots(1,1,figsize=(5*3,3.8*3))
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
    global num_batches_per_epoch
    f = open(logfile)
    losses = []
    times = []
    average_delays = []
    lrs = []
    i = 0
    time0 = None 
    max_epochs = global_max_epochs
    counter = 0
    for line in f.readlines():
        #if line.find('average forward') > 0:
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
        valid = line.find('val acc: ') > 0 if isacc else line.find('average loss: ') > 0
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
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

def plot_loss(logfile, label, isacc=False, title='ResNet-20', fixed_color=None):
    losses, times, average_delays, lrs = read_losses_from_log(logfile, isacc=isacc)
    norm_means, norm_stds = read_norm_from_log(logfile)

    print('times: ', times)
    print('losses: ', losses)
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
        ax.set_ylabel('top-1 Validation Accuracy')
    else:
        ax.set_ylabel('training loss')
    #plt.title('ResNet-50')
    ax.set_title(get_real_title(title))
    marker = markeriter.next()
    if fixed_color:
        color = fixed_color
    else:
        color = coloriter.next()
    #print('marker: ', marker)
    #ax.plot(losses[0:180], label=label, marker=marker, markerfacecolor='none')
    

    if USE_BATCH_X:
        size = int(100000 * global_density)
        iterations = np.arange(len(losses)) 
        iterations *= size 
    else:
        iterations = np.arange(len(losses)) 
    #ax.plot(losses, label=label, marker=marker, markerfacecolor='none', color=color)
    line = ax.plot(iterations, losses, label=label, marker=marker, markerfacecolor='none', color=color, linewidth=1)
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)
    if USE_BATCH_X:
        ax.set_xlabel('# iterations')
    else:
        ax.set_xlabel('# of epochs')
    #plt.plot(times, losses, label=label, marker=markeriter.next())
    #plt.xlabel('Time [s]')
    #ax.grid(linestyle=':')
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
    return line

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
    global global_density
    global_density = density
    postfix='5922'
    color = None
    if prefix.find('allreduce')>=0:
        postfix='0'
    elif prefix.find('single') >= 0:
        postfix = None
    if sparsity:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (prefix, dnn, nworkers, bs, lr, sparsity)
    elif nsupdate:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (prefix, dnn, nworkers, bs, lr, nsupdate)
    else:
        logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f' % (prefix, dnn, nworkers, bs, lr)
    if sg is not None:
        logfile += '-sg%.2f' % sg
    if density is not None:
        logfile += '-ds%s' % str(density)
        #color = fixed_colors['gTopK']
        color = fixed_colors[density]
    else:
        color = fixed_colors['S-SGD']
    if postfix is None:
        logfile += '/%s.log' % (hostname)
    else:
        logfile += '/%s-%s.log' % (hostname, postfix)
    print('logfile: ', logfile)
    if force_legend:
        l = legend
    else:
        l = legend+ '(lr=%.4f, bs=%d, %d workers)'%(lr, bs, nworkers)
    line = plot_loss(logfile, l, isacc=isacc, title=dnn, fixed_color=color) 
    return line

def plot_group_norm_diff():
    global ax
    #networks = ['vgg16', 'resnet20', 'lstm', 'lstman4']
    #networks = ['vgg16', 'resnet20', 'alexnet', 'resnet50', 'lstm', 'lstman4']
    networks = ['resnet20']
    for i, network in enumerate(networks):
        ax_row = i / NFPERROW
        ax_col = i % NFPERROW
        ax = group_axs[ax_row][ax_col]
        ax1 = group_axtwins[ax_row][ax_col]
        plts = plot_norm_diff(ax1, network)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, ncol=4, loc='upper center', fontsize=FONTSIZE, frameon=True)
    #plt.subplots_adjust(bottom=0.09, left=0.08, right=0.90, top=0.88, wspace=0.49, hspace=0.42)
    if NFIGURES == 6:
        plt.subplots_adjust(bottom=0.07, left=0.08, right=0.90, top=0.9, wspace=0.58, hspace=0.49)
    else:
        plt.subplots_adjust(bottom=0.10, left=0.08, right=0.90, top=0.86, wspace=0.58, hspace=0.49)
    plt.savefig('%s/multiple_normdiff.pdf'%OUTPUTPATH)

def plot_norm_diff(lax=None, network=None, subfig=None):
    global global_index
    global global_max_epochs
    density = 0.001
    nsupdate=1
    if network == 'lstm':
        #network = 'lstm';bs =100;lr=1.0;epochs = 80;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-norm'
        #network = 'lstm';bs =32;lr=1.0;epochs =40;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2'
        #network = 'lstm';bs =32;lr=1.0;epochs =40;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug3'
        network = 'lstm';bs =100;lr=30.0;epochs =40;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu2norm'
    elif network == 'lstman4':
        #network = 'lstman4';bs =8;lr=0.0003;epochs = 80;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2'
        network = 'lstman4';bs =8;lr=0.0002;epochs = 80;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2'
        #network = 'lstman4';bs =4;lr=0.0003;epochs = 40;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2'
    elif network == 'resnet20':
        network = 'resnet20';bs =32;lr=0.1;epochs=140;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-normtest'
        #network = 'resnet20';bs =128;lr=0.1;epochs=150;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug'
    elif network == 'vgg16':
        network = 'vgg16';bs=128;lr=0.1;epochs=140;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-normtest'
    elif network == 'alexnet':
        network = 'alexnet';bs=256;lr=0.01;epochs =40;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-norm'
    elif network == 'resnet50':
        nsupdate=16
        network = 'resnet50';bs=512;lr=0.01;epochs =35;prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-norm';
    global_max_epochs = epochs
    path = LOGHOME+'/%s/%s-n4-bs%d-lr%.4f-ns%d-sg1.50-ds%s' % (prefix, network,bs,lr, nsupdate,density)
    print(network, path)
    plts = []
    if network == 'lstm':
        #line = plot_with_params(network, 4, 32, lr, 'host144', r'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, force_legend=True)
        line = plot_with_params(network, 4, 100, 30.0, 'hpclgpu', r'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu3fixedlr', nsupdate=1, force_legend=True)
        plts.append(line)
        #line = plot_with_params(network, 4, bs, lr, 'gpu20', r'gTop-$k$ S-SGD loss', prefix=prefix, nsupdate=1, sg=1.5, density=density, force_legend=True)
        line = plot_with_params(network, 4, 100, 30.0, 'hpclgpu', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu3fixedlr', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    elif network == 'resnet20':
        line = plot_with_params(network, 4, 32, lr, 'gpu21', 'S-SGD loss', prefix='allreduce', force_legend=True)
        #line = plot_with_params(network, 4, 128, lr, 'gpu21', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, force_legend=True)
        plts.append(line)
        line = plot_with_params(network, 4, bs, lr, 'gpu13', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #line = plot_with_params(network, 4, 128, lr, 'gpu21', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plts.append(line)
        pass
    elif network == 'vgg16':
        line = plot_with_params(network, 4, bs, lr, 'gpu13', 'S-SGD loss', prefix='allreduce-baseline-dc1-model-icdcs', nsupdate=1, force_legend=True)
        plts.append(line)
        line = plot_with_params(network, 4, bs, lr, 'gpu10', r'gTop-$k$ S-SGD loss', prefix=prefix, nsupdate=1, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    elif network == 'lstman4':
        #line = plot_with_params(network, 4, 8, 0.0003, 'gpu17', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 8, 0.0003, 'host144', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, force_legend=True)
        line = plot_with_params(network, 4, 8, 0.0002, 'host144', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        plts.append(line)
        #line = plot_with_params(network, 4, 8, 0.0002, 'host144', 'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, sg=1.5, density=density, force_legend=True)
        line = plot_with_params(network, 4, 8, 0.0002, 'gpu17', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #line = plot_with_params(network, 4, bs, lr, 'gpu13', r'gTop-$k$ S-SGD loss', prefix=prefix, nsupdate=1, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    elif network == 'resnet50':
        plot_loss('baselinelogs/accresnet50-lr0.01-c40,70.log', 'S-SGD', isacc=False, title='ResNet-50', fixed_color=fixed_colors['ssgd']) 
        line = plot_with_params(network, 4, 512, lr, 'gpu15', r'gTop-$k$ S-SGD loss', prefix=prefix, nsupdate=nsupdate, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    elif network == 'alexnet':
        plot_with_params(network, 8, 256, 0.01, 'gpu20', 'S-SGD', prefix='allreduce-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, force_legend=True)
        line = plot_with_params(network, 4, 256, lr, 'gpu18', r'gTop-$k$ S-SGD loss', prefix=prefix, nsupdate=nsupdate, sg=1.5, density=density, force_legend=True)
        plts.append(line)
    arr = []
    arr2 = []
    arr3 = []
    for i in range(1, epochs+1):
        fn = '%s/gtopknorm-rank0-epoch%d.npy' % (path, i)
        fn2 = '%s/randknorm-rank0-epoch%d.npy' % (path, i)
        #fn3 = '%s/upbound-rank0-epoch%d.npy' % (path, i)
        #fn4 = '%s/densestd-rank0-epoch%d.npy' % (path, i)
        arr.append(np.mean(np.power(np.load(fn), 2)))
        arr2.append(np.mean(np.power(np.load(fn2), 2)))
        #if arr is None:
        #    arr = np.mean(np.load(fn))
        #    arr2 = np.mean(np.load(fn2))
        #    #arr3 = np.load(fn3)
        #    #arr4 = np.load(fn4)
        #else:
        #    arr = np.concatenate((arr, np.mean(np.load(fn))))
        #    arr2 = np.concatenate((arr2, np.mean(np.load(fn2))))
        #    #arr3 = np.concatenate((arr3, np.load(fn3)))
        #    #arr4 = np.concatenate((arr4, np.load(fn4)))
    arr = np.array(arr)
    arr2 = np.array(arr2)
    cax = lax if lax is not None else ax1
    #ax1.plot(arr-arr2, label='||x-gTopK(x)||-||x-randomK(x)||')
    #cax.plot(arr-arr2, label=r'$\delta$')
    cax.plot(arr/arr2, label=r'$\delta$', color=fixed_colors['blue'],linewidth=1)
    cax.set_ylim(bottom=0.97, top=1.001)
    zero_x = np.arange(len(arr), step=1)
    #cax.plot(zero_x, np.zeros_like(zero_x), ':', label='0 ref.', color='black')
    ones = np.ones_like(zero_x)
    cax.plot(zero_x, ones, ':', label='1 ref.', color='black', linewidth=1)
    if True or network.find('lstm') >= 0:
        subaxes = inset_axes(cax,
                            width='50%', 
                            height='30%', 
                            bbox_to_anchor=(-0.04,0,1,0.95),
                            bbox_transform=cax.transAxes,
                            loc='upper right')
        half = epochs //2
        subx = np.arange(half, len(arr))
        subaxes.plot(subx, (arr/arr2)[half:], color=fixed_colors['blue'], linewidth=1)
        subaxes.plot(subx, ones[half:], ':', color='black', linewidth=1)
        subaxes.set_ylim(bottom=subaxes.get_ylim()[0])
        u.update_fontsize(subaxes, FONTSIZE)
    #ax.plot(arr4, label='Gradients std')
    cax.set_xlabel('# of iteration')
    cax.set_ylabel(r'$\delta$')
    #cax.set_title(STANDARD_TITLES[network])
    #plt.ylim(top=1.)
    #cax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #cax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #ax1.legend(fontsize=FONTSIZE-2, loc='center left')
    #ax.legend(fontsize=FONTSIZE-2, loc='center right')
    u.update_fontsize(cax, FONTSIZE)
    if global_index is not None:
        global_index += 1
    return plts
    #plt.savefig('%s/%s_normdiff.pdf' % (OUTPUTPATH, network))

def sensivities(network):
    if network == 'resnet20':
        plot_with_params(network, 4, 32, 0.1, 'gpu21', 'Dense Gradients (BS=128,lr=0.1)', prefix='allreduce', force_legend=True)
        #plot_with_params(network, 4, 128, 0.1, 'gpu13', 'Dense Gradients (BS=512)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, force_legend=True)
        #plot_with_params(network, 4, 256, 0.1, 'hpclgpu', 'Dense Gradients (BS=1024)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, force_legend=True)
        density=0.001
        plot_with_params(network, 4, 32, 0.1, 'gpu13', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.001
        #plot_with_params(network, 4, 128, 0.1, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=512)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #density=0.0005
        #plot_with_params(network, 4, 32, 0.1, 'gpu10', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0001
        plot_with_params(network, 4, 32, 0.1, 'gpu15', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0001
        #plot_with_params(network, 4, 32, 0.01, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.01)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0001
        plot_with_params(network, 4, 128, 0.1, 'gpu14', r'gTop-$k$ with $\rho=%s$ (BS=512)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plot_with_params(network, 4, 128, 0.1, 'gpu13', r'gTop-$k$ with $\rho=%s$ (BS=512, no warmup)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.0316, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.0316)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plot_with_params(network, 4, 32, 0.035, 'gpu13', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.035)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plot_with_params(network, 4, 32, 0.035, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.035)-ll2'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-ll2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.035, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.035)-ll'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-ll', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.0448, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.0448)-ll'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-ll', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.05, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.05)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.08, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.08)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0001
        #plot_with_params(network, 4, 128, 0.1, 'gpu14', r'gTop-$k$ with $\rho=%s$ (BS=512)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
    elif network == 'vgg16':
        plot_with_params(network, 4, 32, 0.1, 'gpu13', 'Dense Gradients (BS=128)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 128, 0.1, 'gpu13', 'Dense Gradients (BS=512)', prefix='allreduce-baseline-dc1-model-icdcs', nsupdate=1, force_legend=True)
        density=0.001
        #plot_with_params(network, 4, 128, 0.1, 'gpu10', r'gTop-$k$ with $\rho=%s$'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0005
        #plot_with_params(network, 4, 128, 0.1, 'gpu20', r'gTop-$k$ with $\rho=%s$'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plot_with_params(network, 4, 128, 0.1, 'gpu10', r'gTop-$k$ with $\rho=%s$ (BS=512)'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0005
        plot_with_params(network, 4, 128, 0.1, 'gpu20', r'gTop-$k$ with $\rho=%s$ (BS=512)'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0001
        plot_with_params(network, 4, 128, 0.1, 'gpu20', r'gTop-$k$ with $\rho=%s$ (BS=512)'%str(density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density=0.0001
        plot_with_params(network, 4, 32, 0.1, 'gpu13', r'gTop-$k$ with $\rho=%s$ (BS=128)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
    plt.legend(fontsize=FONTSIZE)

def lr_sensivities(network):
    global global_max_epochs 
    if network == 'resnet20':
        #plot_with_params(network, 4, 32, 0.1, 'gpu21', 'Dense Gradients (BS=128,lr=0.1)', prefix='allreduce', force_legend=True)
        plot_with_params(network, 4, 32, 0.1, 'gpu13', r'Dense Gradients (BS=128,lr=0.1)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-fixedlr', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 32, 0.0035, 'hpclgpu', r'Dense Gradients (BS=128,lr=0.0035)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-fixedlr', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 32, 0.035, 'gpu11', r'Dense Gradients (BS=128,lr=0.035)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-fixedlr', nsupdate=1, force_legend=True)
        density = 0.0001
        plot_with_params(network, 4, 32, 0.1, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plot_with_params(network, 4, 32, 0.01, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.01)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plot_with_params(network, 4, 32, 0.035, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.035)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr', nsupdate=1, sg=1.5, density=density, force_legend=True)
        plot_with_params(network, 4, 32, 0.0035, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.0035)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr', nsupdate=1, sg=1.5, density=density, force_legend=True)
    elif network == 'lstman4':
        density = 0.001
        #line = plot_with_params(network, 4, 4, 0.0003, 'gpu17', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 8, 0.0003, 'host144', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 8, 0.0002, 'host144', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        line = plot_with_params(network, 4, 32, 0.0003, 'gpu20', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 8, 0.0003, 'gpu16', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 8, 0.0003, 'host144', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 4, 0.0003, 'gpu17', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #line = plot_with_params(network, 4, 8, 0.0003, 'gpu13', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #line = plot_with_params(network, 4, 8, 0.0002, 'gpu17', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #line = plot_with_params(network, 4, 32, 0.0003, 'gpu20', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, sg=1.5, density=density, force_legend=True)
        line = plot_with_params(network, 4, 32, 0.0003, 'gpu16', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu2', nsupdate=1, sg=1.5, density=density, force_legend=True)
    elif network == 'resnet50':
        #plot_with_params(network, 8, 64, 0.01, 'gpu10', 'S-SGD 8 workers', prefix='allreduce-debug', force_legend=True)
        plot_loss('baselinelogs/accresnet50-lr0.01-c40,70.log', 'S-SGD', isacc=False, title='ResNet-50') 
        density = 0.001
        line = plot_with_params(network, 4, 512, 0.01, 'gpu15', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-norm', nsupdate=16, sg=1.5, density=density, force_legend=True)
    elif network == 'alexnet':
        plot_with_params(network, 8, 256, 0.01, 'gpu20', 'S-SGD', prefix='allreduce-baseline-gwarmup-dc1-model-icdcs-n-profiling', nsupdate=1, force_legend=True)
        density = 0.001
        line = plot_with_params(network, 4, 256, 0.01, 'gpu18', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-norm', nsupdate=1, sg=1.5, density=density, force_legend=True)
    elif network == 'lstm':
        global_max_epochs = 40
        #line = plot_with_params(network, 4, 32, 1.0, 'host144', r'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 32, 1.0, 'gpu17', r'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-debug2', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 8, 20.0, 'gpu16', r'S-SGD loss(BS=32)fx', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu2fixedlr', nsupdate=1, force_legend=True)
        line = plot_with_params(network, 4, 5, 20.0, 'gpu14', r'S-SGD loss(BS=20)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        #line = plot_with_params(network, 4, 32, 20.0, 'host144', r'S-SGD loss(BS=128)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu2', nsupdate=1, force_legend=True)
        density = 0.001
        #line = plot_with_params(network, 4, 32, 20.0, 'hpclgpu', r'gTop-$k$ S-SGD loss(BS=128,hpclgpu)', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #line = plot_with_params(network, 4, 32, 20.0, 'host144', r'gTop-$k$ S-SGD loss(BS=128,host144)', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu2', nsupdate=1, sg=1.5, density=density, force_legend=True) # not good
        line = plot_with_params(network, 4, 8, 20.0, 'gpu14', r'gTop-$k$ S-SGD loss(BS=32)', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #line = plot_with_params(network, 4, 32, 20.0, 'gpu20', r'gTop-$k$ S-SGD loss(BS=128,gpu20)', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        line = plot_with_params(network, 4, 8, 20.0, 'host144', r'gTop-$k$ S-SGD loss(BS=32,host144)', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, sg=1.5, density=density, force_legend=True)
        line = plot_with_params(network, 4, 5, 20.0, 'gpu14', r'gTop-$k$ S-SGD loss(BS=20)', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu2fixedlr', nsupdate=1, sg=1.5, density=density, force_legend=True)

    plt.legend(fontsize=FONTSIZE)

def atopk_sensivities(network):
    if network == 'resnet20':
        plot_with_params(network, 4, 32, 0.1, 'gpu21', 'Dense Gradients (BS=128,lr=0.1)', prefix='allreduce', force_legend=True)
        density = 0.0001
        #plot_with_params(network, 4, 32, 0.1, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density), prefix='allreduce-comp-atopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.001
        plot_with_params(network, 4, 32, 0.1, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density), prefix='allreduce-comp-atopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.01
        plot_with_params(network, 4, 32, 0.1, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density), prefix='allreduce-comp-atopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.1, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai', nsupdate=1, sg=1.5, density=density, force_legend=True)
    plt.legend(fontsize=FONTSIZE)

def new_lrs():
    #base_s = 1000
    #lrs = {base_s: 0.1}
    base_s = 200 
    lrs = {base_s: 1.0}
    ss = [1000, 2000, 4000, 10000, 15000, 20000, 100000]
    new_lrs = []
    for s in ss:
        new_lr = np.sqrt(np.power(base_s/float(s), 3)) * lrs[base_s]
        lrs[s] = new_lr
        new_lrs.append(new_lr)
    print(1.0/np.array(ss))
    print(new_lrs)


def lr_sensivities_hpclgpu(network):
    if network == 'resnet20':
        #plot_with_params(network, 4, 32, 0.1, 'hpclgpu', r'Dense Gradients (BS=128,lr=0.1)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, force_legend=True)
        density = 0.001;legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 32, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.00025;legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 32, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.0125, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.0125)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.0001;legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density) # This is can also be used
        plot_with_params(network, 4, 32, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.0032, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.0032)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.00005;legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density) # This is can also be used
        plot_with_params(network, 4, 32, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.0011, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.0011)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #density = 0.00001;legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$s=%d$'%(1/density) # Should not be used either!
        #plot_with_params(network, 4, 32, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 0.0001, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.0001)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
    elif network == 'vgg16':
        #plot_with_params(network, 4, 128, 0.1, 'hpclgpu', r'Dense Gradients (BS=512,lr=0.1)', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, force_legend=True)
        density = 0.001;legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 128, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.00025
        legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 128, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 128, 0.0125, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=512,lr=0.0125)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.0001
        legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 128, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 128, 0.00316, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=512,lr=0.00316)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.00005
        legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 128, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 128, 0.00112, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=512,lr=0.00112)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #density = 0.00001
        #plot_with_params(network, 4, 128, 0.1, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=512,lr=0.1)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 128, 0.0001, 'hpclgpu', r'gTop-$k$ with $\rho=%s$ (BS=512,lr=0.0001)'%str(density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
    elif network == 'lstm':
        line = plot_with_params(network, 4, 32, 40, 'hpclgpu', 'S-SGD loss', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        density = 0.001
        line = plot_with_params(network, 4, 32, 40, 'hpclgpu', r'gTop-$k$ S-SGD loss', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2', nsupdate=1, sg=1.5, density=density, force_legend=True)


    plt.legend(prop={'family':FONT_FAMILY, 'size':FONTSIZE})
    plt.subplots_adjust(bottom=0.174, left=0.159, right=0.963, top=0.902, wspace=0.2, hspace=0.2)

def lr_sensivities_host144(network):
    global global_max_epochs
    if network == 'lstman4':
        global_max_epochs = 80
        density = 0.001; legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        line = plot_with_params(network, 4, 8, 0.0003, 'gpu13', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        line = plot_with_params(network, 4, 8, 0.0002, 'gpu13', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.00025; legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 8, 0.0003, 'host144', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.0001; legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 8, 0.0003, 'host144', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        density = 0.00005; legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        plot_with_params(network, 4, 8, 0.0003, 'host144', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #density = 0.00001; legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$s$=%d'%(1/density)
    elif network == 'lstm':
        plot_with_params(network, 1, 20, 20.0, 'host144', r'S-SGD', prefix='singlegpu-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 32, 2.0, 'host144', r'S-SGD', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        plot_with_params(network, 4, 8, 20.0, 'host144', r'S-SGD', prefix='allreduce-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, force_legend=True)
        density = 0.001
        plot_with_params(network, 4, 32, 2.0, 'host144', r'gTop-$k$ $c$=%d'%int(1/density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #plot_with_params(network, 4, 32, 1.0, 'gpu11', r'gTop-$k$ %d'%int(1/density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, sg=1.5, density=density, force_legend=True)
        #density = 0.00001; legend=r'gTop-$k$ with $\rho=%s$ (BS=128,lr=0.1)'%str(density);legend=r'$c$=%d'%(1/density)
        #plot_with_params(network, 4, 8, 0.0003, 'host144', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
    plt.legend(prop={'family':FONT_FAMILY, 'size':FONTSIZE})
    plt.subplots_adjust(bottom=0.174, left=0.159, right=0.963, top=0.902, wspace=0.2, hspace=0.2)

def plot_group_lr_sensitivies():
    def _plot_with_network(network):
        global global_max_epochs
        global global_density
        densities = [0.001, 0.00025, 0.0001, 0.00005]#, 0.00001]
        if network == 'vgg16':
            global_max_epochs = 140
            for density in densities:
                legend=r'$c$=%d'%(1/density)
                plot_with_params(network, 4, 128, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)

        elif network == 'resnet20':
            global_max_epochs = 140
            for density in densities:
                legend=r'$c$=%d'%(1/density)
                plot_with_params(network, 4, 32, 0.1, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd', nsupdate=1, sg=1.5, density=density, force_legend=True)
        elif network == 'lstm':
            global_max_epochs = 40
            for density in densities:
                legend=r'$c$=%d'%(1/density)
                if density == 0.001:
                    plot_with_params(network, 4, 100, 30.0, 'hpclgpu', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu3fixedlr', nsupdate=1, sg=1.5, density=density, force_legend=True)
                else:
                #plot_with_params(network, 4, 100, 1.0, 'gpu11', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlrfx', nsupdate=1, sg=1.5, density=density, force_legend=True)
                #plot_with_params(network, 4, 32, 1.0, 'host144', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, sg=1.5, density=density, force_legend=True)
                    plot_with_params(network, 4, 100, 30.0, 'host144', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu4lr', nsupdate=1, sg=1.5, density=density, force_legend=True)
        elif network == 'lstman4':
            global_max_epochs = 80
            for density in densities:
                legend=r'$c$=%d'%(1/density)
                #if density == 0.001:
                #    host = 'gpu13';prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-debug2'
                #else:
                #    host = 'host144';prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-fixedlr2nd'
                #plot_with_params(network, 4, 8, 0.0003, host, legend, prefix=prefix, nsupdate=1, sg=1.5, density=density, force_legend=True)
                plot_with_params(network, 4, 8, 0.0002, 'host144', legend, prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-ijcai-wu1', nsupdate=1, sg=1.5, density=density, force_legend=True)
    global ax
    networks = ['vgg16', 'resnet20', 'lstm', 'lstman4']
    for i, network in enumerate(networks):
        ax_row = i / NFPERROW
        ax_col = i % NFPERROW
        ax = group_axs[ax_row][ax_col]
        _plot_with_network(network)
        ax.legend(ncol=1, loc='upper right', fontsize=FONTSIZE-2)
    #lines, labels = ax.get_legend_handles_labels()
    #fig.legend(lines, labels, ncol=4, loc='upper center', fontsize=FONTSIZE)
    #plt.subplots_adjust(bottom=0.10, left=0.10, right=0.94, top=0.95, wspace=0.37, hspace=0.39)
    plt.subplots_adjust(bottom=0.10, left=0.10, right=0.94, top=0.95, wspace=0.37, hspace=0.42)
    plt.savefig('%s/multiple_lrs.pdf'%OUTPUTPATH)


if __name__ == '__main__':
    network = 'resnet20'
    #network = 'lstman4'
    #network = 'lstm'
    #network = 'resnet50'
    #network = 'alexnet'
    
    if NFIGURES == 1:
        lr_sensivities(network)
        #lr_sensivities_host144(network)
        #lr_sensivities_hpclgpu(network)
    else:
        if PLOT_NORM:
            plot_group_norm_diff()
        else:
            plot_group_lr_sensitivies()
    #sensivities(network)
    #lr_sensivities_hpclgpu(network)
    #atopk_sensivities(network)
    plt.show()
    #new_lrs()

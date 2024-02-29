# -*- coding: utf-8 -*-
from __future__ import print_function
import time
from matplotlib import rcParams
#FONT_FAMILY='DejaVu Serif'
#rcParams["font.family"] = FONT_FAMILY 
from mpl_toolkits.axes_grid.inset_locator import inset_axes, zoomed_inset_axes
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
#matplotlib.rcParams['pdf.fonttype'] = 42
#matplotlib.rcParams['ps.fonttype'] = 42
from plot_sth import Bar
import plot_sth as Color
#rc('text', usetex=True)
matplotlib.use("TkAgg")
import numpy as np
import datetime
import itertools
import utils as u
import readers
PLOT_TYPE = 'NORM'
#PLOT_TYPE = 'THROUGHPUT'
#PLOT_TYPE = 'ACC'
PLOT_TYPE = 'OTHERS'
#PLOT_TYPE = 'NONE'

#markers=['.','x','o','v','^','<','>','1','2','3','4','8','s','p','*']
markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
delta_colors = ['#F1948A', '#C0392B', '#9B59B6', '#2980B9', '#3498DB', '#17A589', '#229954', '#D4AC0D', '#D4AC0D', '#D68910', '#A6ACAF', '#2E4053', '#5B33FF', '#E72913']
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)

fixed_markers = {
        'lgs loss': '+',
        'slgs loss': 'x',
        'loss':'+',
        }

fixed_colors = {
        #'S-SGD': 'r',
        #'gTopK': 'g'
        'dense-sgd':'r',
        'slgs-sgd':'blue',
        'slgs loss':'blue',
        'loss':'g',
        'lgs-sgd':'g',
        'lags-sgd':'g',
        'lgs loss':'g',
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
STANDARD_MARKERS = {
        'dense':'o',
        'dense-sgd':'o',
        'topk':'x',
        'slgs-sgd':'x',
        'top-k':'x',
        'gtopk':'^',
        'lgs-sgd':'^',
        'lags-sgd':'^',
        'gtop-k':'^',
        }


#jOUTPUTPATH='/media/sf_Shared_Data/tmp/iclr2020'
OUTPUTPATH='/home/shshi/gpuhome/plots/posticlr'
LOGHOME='/media/sf_Shared_Data/gpuhome/repositories/p2p-dl/logs'
LOGHOME='/home/shshi/work/p2p-dl/logs'

EPOCH = True
FONTSIZE=16
num_batches_per_epoch = None
global_max_epochs=150
global_density=0.001
USE_BATCH_X = False

PLOT_NORM=False
USE_TIME=False
#PLOT_NORM=True
if PLOT_TYPE == 'NORM':
    PLOT_NORM = True
    #NFIGURES=4;NFPERROW=2
    NFIGURES=3;NFPERROW=3
    if NFIGURES ==3:
        FIGSIZE=(4.2*NFPERROW,3.2)
    else:
        FIGSIZE=(5.2*NFPERROW,3*NFIGURES/NFPERROW)
elif PLOT_TYPE == 'ACC':
    USE_TIME=False
    #NFIGURES=4;NFPERROW=2
    #FIGSIZE=(4.5*NFPERROW,3.0*NFIGURES/NFPERROW)

    NFIGURES=3;NFPERROW=3
    if NFIGURES ==3:
        FIGSIZE=(3.8*NFPERROW,3.2)
    else:
        FIGSIZE=(5.2*NFPERROW,3*NFIGURES/NFPERROW)

elif PLOT_TYPE == 'THROUGHPUT':
    #NFIGURES=1;NFPERROW=1
    #FIGSIZE=(4.5*NFPERROW,3.0*NFIGURES/NFPERROW)

    NFIGURES=3;NFPERROW=3
    #NFIGURES=1;NFPERROW=1
    if NFIGURES ==3:
        FIGSIZE=(3.8*NFPERROW,3.2)
    else:
        FIGSIZE=(4.5*NFPERROW,3.0*NFIGURES/NFPERROW)
elif PLOT_TYPE == 'OTHERS':
    NFIGURES=1;NFPERROW=1
    FIGSIZE=(4.5*NFPERROW,3.2)

if PLOT_TYPE == 'NONE':
    pass
else:
    fig, group_axs = plt.subplots(NFIGURES/NFPERROW, NFPERROW,figsize=FIGSIZE)
    print('group_axs: ', group_axs)
    if NFIGURES > 1 and PLOT_NORM:
        ax = None
        group_axtwins = []
        num_rows = NFIGURES/NFPERROW 
        for i in range(num_rows):
            tmp = []
            if num_rows > 1:
                for a in group_axs[i]:
                    tmp.append(a.twinx())
                group_axtwins.append(tmp)
            else:
                for a in group_axs:
                    tmp.append(a.twinx())
                group_axtwins = tmp
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
        'resnet20': 'ResNet-20 on CIFAR10',
        'resnet110': 'ResNet-110',
        'vgg16': 'VGG-16 on CIFAR10',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50 in ImageNet',
        'inceptionv4': 'Inception-v4',
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
        valid = line.find('val top-1 acc: ') > 0 or line.find('val acc:') > 0 if isacc else line.find('avg loss: ') > 0
        if line.find('Epoch') > 0 and valid: 
            items = line.split(' ')
            if isacc:
                if line.find('val top-1 acc: ') > 0:
                    loss = float(items[-4][0:-1])
                else:
                    loss = float(items[-1][0:-1]) * 100
            else:
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
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
        valid = line.find('val top-1 acc: ') > 0 or line.find('val acc:') > 0 if isacc else line.find('avg loss: ') > 0
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
        if line.find('Epoch') > 0 and valid:
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
    if isacc:
        print('max validation accuracy: ', np.max(losses) if len(losses) > 0 else 0.0)
        if logfile.find('lstm') > 0:
            print('min ppl: ', np.min(losses))


    if len(average_delays) > 0:
        delay = int(np.mean(average_delays))
    else:
        delay = 0
    if delay > 0:
        label = label + ' (delay=%d)' % delay
    if isacc:
        ax.set_ylabel('val accuracy')
        if logfile.find('lstm') > 0:
            ax.set_ylabel('perplexity')
            
    else:
        ax.set_ylabel('training loss')
    #ax.set_title(get_real_title(title))
    #marker = fixed_markers.get(label.lower(), None) #markeriter.next()
    marker = markeriter.next()
    if fixed_color:
        color = fixed_color
    else:
        color = coloriter.next()
    if USE_BATCH_X:
        size = int(100000 * global_density)
        iterations = np.arange(len(losses)) 
        iterations *= size 
    else:
        iterations = np.arange(len(losses)) 
    if USE_TIME:
        line = ax.plot(times, losses, label=label, marker=marker, markerfacecolor='none', color=color, linewidth=1)
    else:
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
        ax.set_xlabel('epochs')
    if USE_TIME:
        ax.set_xlabel('train time [sec]')
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
            label = host+' ('+str(nworkers)+' workers)'
            if baseline:
                label += ' Baseline'
            plot_loss(logfile, label) 

def plot_with_params(dnn, nworkers, bs, lr, hostname, legend, isacc=False, prefix='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False, force_color=None):
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
        #color = fixed_colors[density]
    else:
        color = fixed_colors['S-SGD']
    if force_color:
        color = force_color
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
    global LOGHOME
    global FONTSIZE
    FONTSIZE = 12
    LOGHOME='/media/sf_Shared_Data/gpuhome/repositories/p2p-dl/logs'
    #networks = 'vgg16', 'resnet20',
    networks = ['resnet20', 'vgg16', 'lstm'] #, 'lstman4']
    #networks = ['vgg16', 'resnet20', 'alexnet', 'resnet50', 'lstm', 'lstman4']
    num_rows = NFIGURES/NFPERROW
    for i, network in enumerate(networks):
        ax_row = i / NFPERROW
        ax_col = i % NFPERROW
        if num_rows > 1:
            ax = group_axs[ax_row][ax_col]
            ax1 = group_axtwins[ax_row][ax_col]
        else:
            ax = group_axs[i]
            ax1 = group_axtwins[i]
        if network == 'vgg16':
            axd = ax1
        plts = plot_norm_diff(ax1, network)
        ax.grid(linestyle=':')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = axd.get_legend_handles_labels()
    if num_rows == 1:
        fig.legend(lines + lines2, labels + labels2, ncol=len(lines)+len(lines2), loc='upper center', fontsize=12, frameon=True)
    else:
        fig.legend(lines + lines2, labels + labels2, ncol=1, loc='center right', fontsize=FONTSIZE, frameon=True)
    if NFIGURES == 4:
        plt.subplots_adjust(bottom=0.10, left=0.07, right=0.8, top=0.94, wspace=0.49, hspace=0.44)
    else:
        plt.subplots_adjust(bottom=0.17, left=0.05, right=0.94, top=0.76, wspace=0.52, hspace=0.44)
    plt.savefig('%s/multiple_normdiff.pdf'%OUTPUTPATH)

def plot_norm_diff(lax=None, network=None, subfig=None):
    global global_index
    global global_max_epochs
    density = 0.001
    nsupdate=1
    plts = []
    if network == 'lstm':
        global_max_epochs=40;bs=100;lr=1.0;density=0.001
        nworkers = 16; gpu='gpu15'
        line = plot_with_params(network, nworkers, bs, lr, gpu, r'loss', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-norm', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
        plts.append(line)
    elif network == 'resnet20':
        #nworkers=32; gpu='gpu13'
        nworkers=16; gpu='gpu13'
        #nworkers=4; gpu='gpu19'
        global_max_epochs=120;bs=32;lr=0.1;density=0.001
        line = plot_with_params(network, nworkers, bs, lr, gpu, r'loss', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-norm', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
        #line = plot_with_params(network, nworkers, bs, lr, 'gpu10', r'SLGS loss', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-norm', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
        plts.append(line)
    elif network == 'vgg16':
        global_max_epochs=120;bs=128;lr=0.1;density=0.001
        #nworkers=4; gpu='gpu15'
        nworkers = 16; gpu='gpu10'
        #nworkers = 32; gpu='gpu13'
        line = plot_with_params(network, nworkers, bs, lr, gpu, r'loss', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-norm', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
        #line = plot_with_params(network, nworkers, bs, lr, 'gpu10', r'SLGS loss', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-norm', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
        plts.append(line)
    elif network == 'lstman4':
        global_max_epochs=80;bs=4;lr=0.0002;density=0.001
        nworkers = 16; gpu='gpu10'
        line = plot_with_params(network, nworkers, bs, lr, gpu, r'loss', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-norm', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='g')
    elif network == 'resnet50':
        pass
    elif network == 'alexnet':
        pass
    path = './logs/allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-norm/%s-n%d-bs%d-lr%.4f-ns%d-sg2.50-ds%s' % (network,nworkers, bs,lr, 1,str(density))
    #path = './logs/allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-norm/%s-n%d-bs%d-lr%s-ns1-sg2.50-ds0.001/topknorm-rank22-epoch25.npy
    arr = {}; arr2 = {}; deltas = {}
    for i in range(1, global_max_epochs+1):
        fn = '%s/topknorm-rank0-epoch%d.npy' % (path, i)
        data = np.load(fn).item()
        for layer_name in data:
            layer_data = data[layer_name]
            topk = []; randk = []
            for topk_norm, randk_norm, upbound, xnorm, dense_std in layer_data:
                topk.append(topk_norm)
                randk.append(randk_norm)
            topk = np.array(topk)
            randk= np.array(randk)
            topknorm = np.mean(np.power(topk, 2))
            randknorm = np.mean(np.power(randk, 2))
            u.force_insert_item(arr, layer_name, topknorm)
            u.force_insert_item(arr2, layer_name, randknorm)
            u.force_insert_item(deltas, layer_name, topknorm/randknorm)
    keys = deltas.keys()
    layer_name = keys[0]
    current_deltas = deltas[layer_name]
    print('current_deltas: ', keys)
    cax = lax if lax is not None else ax1
    max_layers=7
    zero_x = np.arange(len(current_deltas), step=1)
    ones = np.ones_like(zero_x)
    cax.plot(zero_x, ones, ':', label='1 ref.', color='black', linewidth=1)
    for idx, layer_name in enumerate(keys):
        if idx == max_layers:
            break
        current_deltas = deltas[layer_name]
        cax.plot(current_deltas, label=r'$\delta^{(%d)}$'%(idx+1), color=delta_colors[idx],linewidth=1)
    cax.set_ylim(bottom=0.8, top=1.005)
    if True or network.find('lstm') >= 0:
        subaxes = inset_axes(cax,
                            width='50%', 
                            height='30%', 
                            bbox_to_anchor=(-0.02,-0.06,1,0.95),
                            bbox_transform=cax.transAxes,
                            loc='upper right')
        half = global_max_epochs//2
        for idx, layer_name in enumerate(keys):
            current_deltas = deltas[layer_name]
            subx = np.arange(half, len(current_deltas))
            subaxes.plot(subx, current_deltas[half:], color=delta_colors[idx], linewidth=1)
            subaxes.plot(subx, ones[half:], ':', color='black', linewidth=1)
            if idx > max_layers:
                break
            #subaxes.set_ylim(bottom=subaxes.get_ylim()[0])

        #subx = np.arange(half, len(current_deltas))
        #subaxes.plot(subx, current_deltas[half:], color=fixed_colors['blue'], linewidth=1)
        #subaxes.plot(subx, ones[half:], ':', color='black', linewidth=1)
        u.update_fontsize(subaxes, FONTSIZE)
    cax.set_ylabel(r'$\delta$')
    u.update_fontsize(cax, FONTSIZE)
    #plt.savefig('%s/%s_normdiff.pdf' % (OUTPUTPATH, network))
    if global_index is not None:
        global_index += 1
    return plts


def plot_sub_boxes(ax, name=None):

    #Plot subboxes
    bbox_to_anchor = (-0.02, -0.08, 1, 0.95)
    if name == 'lstm':
        bbox_to_anchor = (-0.02, -0.28, 1, 0.95)
    subaxes = inset_axes(ax,
        width='50%', 
        height='30%', 
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        loc='upper right')
    plts = ax.lines
    #subaxes.set_ylim(bottom=0.85, top=0.95)
    for line in plts:
        x = line.get_xdata()
        y = line.get_ydata()
        half = len(x)*3//4
        subx = x[half:]
        suby = y[half:]
        subaxes.plot(subx, suby, color=line.get_color(), linewidth=1.5)
    subaxes.set_ylim(bottom=subaxes.get_ylim()[0])

def plot_group_convergences():
    global ax
    global FONTSIZE
    global LOGHOME
    #LOGHOME='/home/shshi/work/p2p-dl/logs'
    LOGHOME='./logs'
    FONTSIZE = 14
    #networks = ['resnet20', 'vgg16', 'lstm']#, 'lstm']#, 'lstman4']
    networks = ['resnet20', 'vgg16', 'resnet50']
    num_rows = NFIGURES/NFPERROW
    for i, network in enumerate(networks):
        ax_row = i / NFPERROW
        ax_col = i % NFPERROW
        if NFIGURES/NFPERROW == 1:
            ax = group_axs[i]
        else:
            ax = group_axs[ax_row][ax_col]
        plts = plot_convergence(ax, network)
        ax.grid(linestyle=':')
        plot_sub_boxes(ax, network)
    lines, labels = ax.get_legend_handles_labels()

    if num_rows == 1:
        fig.legend(lines, labels, ncol=4, loc='upper left', fontsize=12, frameon=True)
        plt.subplots_adjust(bottom=0.17, left=0.07, right=0.99, top=0.80, wspace=0.32, hspace=0.49)
    else:
        fig.legend(lines, labels, ncol=4, loc='upper left', fontsize=FONTSIZE, frameon=True)
        plt.subplots_adjust(bottom=0.08, left=0.09, right=0.97, top=0.88, wspace=0.27, hspace=0.49)
    #plt.savefig('%s/group_convergence.pdf'%OUTPUTPATH)


def plot_single_convergence():
    global ax
    global FONTSIZE
    global LOGHOME
    LOGHOME='./logs'
    FONTSIZE=14

    network = 'vgg16'
    #network = 'resnet20'
    #network = 'resnet50'
    plot_convergence(ax, network)
    plot_sub_boxes(ax, network)
    ax.grid(linestyle=':')
    ax.legend(fontsize=FONTSIZE)
    u.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/conv_topkvsrandk_%s.pdf'% (OUTPUTPATH, network), bbox_inches='tight')
    #plt.savefig('%s/conv_gaussiank_%s.pdf'% (OUTPUTPATH, network), bbox_inches='tight')

def plot_convergence(lax=None, network=None, subfig=None):
    global global_index
    global global_max_epochs
    density = 0.001
    plts = []
    isacc=True
    if network == 'lstm':
        global_max_epochs = 40;bs=32;lr=22.0;nworkers=32
        density=1.0
        line = plot_with_params(network, 16, 20, lr, 'gpu16', r'P-SGD', isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-model-nips-layerwise-convlstm-adaptive', nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        density=0.004
    elif network == 'resnet20':
        global_max_epochs = 120;bs=32;lr=0.1; nworkers=16
        density=1.0;compressor='dense'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw227', r'Dense-SGD', isacc=isacc, prefix='allreduce-gwarmup-dc1-model-iclr-convergence-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='black')
        plts.append(line)
        density=0.001;compressor='topk'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'TopK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        #density=0.001;compressor='topk2'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'TopK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='y')
        #plts.append(line)
        density=0.001; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'GaussianK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='g')
        plts.append(line)
        density=0.001; compressor='redsync'
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'RedSync-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-convergence-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='c')
        plts.append(line)
        #density=0.001; compressor='gaussion2'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'GaussianK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='#11920D')
        #plts.append(line)
        #density=0.001; compressor='randomk'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'RandK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='b')
        #plts.append(line)
        #density=0.001; compressor='dgcsampling'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw225', r'DGC-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-posticlr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='c')
        #plts.append(line)
        #density=0.001; compressor='randomksame'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'RandKSame-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='c')
        #plts.append(line)
        #density=0.001; compressor='randomksameec'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'RandKSameEC-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='m')
        plts.append(line)
    elif network == 'vgg16':
        global_max_epochs = 120;bs=128;lr=0.1;nworkers=16
        density=1.0;compressor='dense'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw227', r'Dense-SGD', isacc=isacc, prefix='allreduce-gwarmup-dc1-model-iclr-convergence-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='black')
        plts.append(line)
        density=0.001; compressor='topk'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'TopK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        #density=0.001; compressor='topk2'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'TopK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='y')
        #plts.append(line)
        density=0.001; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'GaussianK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='g')
        plts.append(line)
        #density=0.001; compressor='gaussion2'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'GaussianK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='#11920D')
        #plts.append(line)
        #density=0.001; compressor='randomk'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'RandK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='b')
        #plts.append(line)
        #density=0.001; compressor='dgcsampling'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw225', r'DGC-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-posticlr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='c')
        #plts.append(line)
        density=0.001; compressor='redsync'
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'RedSync-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-convergence-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='c')
        plts.append(line)
        #density=0.001; compressor='randomksame'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'RandKSame-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='c')
        #plts.append(line)
        #density=0.001; compressor='randomksameec'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'RandKSameEC-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='m')
        #plts.append(line)
    elif network == 'resnet110':
        global_max_epochs = 120;bs=128;lr=0.1;nworkers=16
        density=1.0;compressor='dense'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'Dense-SGD', isacc=isacc, prefix='allreduce-gwarmup-dc1-model-iclr-convergence-wu2-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='black')
        plts.append(line)
        density=0.001; compressor='topk'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'TopK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        density=0.001; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'GaussianK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu2-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='g')
        plts.append(line)
        density=0.001; compressor='sigmathresallgather'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'RandomK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='b')
        plts.append(line)
    elif network == 'lstman4':
        pass
    elif network == 'resnet50':
        global_max_epochs = 70;bs=32;lr=0.01; nworkers=16; 
        density=1.0
        line = plot_with_params(network, nworkers, bs, lr, 'hsw224', r'Dense-SGD', isacc=isacc, prefix='allreduce-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='black')
        plts.append(line)
        density=0.001
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'TopK-SGD',isacc=isacc,  prefix='allreduce-comp-topk-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes', nsupdate=1, density=density, force_legend=True,force_color='r')
        plts.append(line)
        #density=0.001; compressor='gaussion'
        #line = plot_with_params(network, nworkers, bs, lr, 'hsw224', r'GaussianK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='g')
        #plts.append(line)
        density=0.001; compressor='randomk'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw215', r'RandK-SGD', isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-accuracy-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='b')
        plts.append(line)
    elif network == 'alexnet':
        global_max_epochs = 120;bs=128;lr=0.01
        pass
    return plts


def plot_single_convergence_sensitivity():
    global ax
    global FONTSIZE
    global LOGHOME
    global global_max_epochs
    LOGHOME='./logs'
    FONTSIZE=14
    isacc = True

    network = 'vgg16'
    network = 'resnet20'
    #network = 'resnet50'
    if network == 'vgg16':
        global_max_epochs = 120;bs=128;lr=0.1;nworkers=16
        density=1.0;compressor='dense'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw227', r'Dense-SGD', isacc=isacc, prefix='allreduce-gwarmup-dc1-model-iclr-convergence-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='black')
        density=0.001; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='g')
        density=0.005; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-convergence-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='blue')
        density=0.01; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-convergence-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='red')
    elif network == 'resnet20':
        global_max_epochs = 120;bs=32;lr=0.1; nworkers=16
        density=1.0;compressor='dense'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw227', r'Dense-SGD', isacc=isacc, prefix='allreduce-gwarmup-dc1-model-iclr-convergence-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='black')
        #plts.append(line)
        density=0.001; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw226', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='g')
        density=0.005; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-convergence-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='blue')
        density=0.01; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-convergence-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='red')
        #plts.append(line)
    elif network == 'resnet50':
        global_max_epochs = 70;bs=32;lr=0.01; nworkers=16; 
        density=1.0
        line = plot_with_params(network, nworkers, bs, lr, 'hsw224', r'Dense-SGD', isacc=isacc, prefix='allreduce-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes', nsupdate=1, density=density, force_legend=True, force_color='black')
        density=0.001; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'hsw224', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-wu3-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='g')
        density=0.005; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-convergence-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='blue')
        density=0.01; compressor='gaussion'
        line = plot_with_params(network, nworkers, bs, lr, 'scigpu10', r'GaussianK-SGD ($k=%.3fd$)'%density, isacc=isacc, prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-convergence-thres-512000kbytes'%compressor, nsupdate=1, density=density, force_legend=True, force_color='red')


    plot_sub_boxes(ax, network)
    ax.grid(linestyle=':')
    ax.legend(fontsize=FONTSIZE-2)
    u.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/sensitivity_gaussiank_%s.pdf'% (OUTPUTPATH, network), bbox_inches='tight')


def plot_others():
    global global_index
    global global_max_epochs
    global LOGHOME
    LOGHOME='/media/sf_Shared_Data/gpuhome/repositories/p2p-dl/logs'
    density = 0.001
    plts = []
    isacc=True
    network='alexnet';global_max_epochs = 120;bs=128;lr=0.01
    density=1.0
    line = plot_with_params(network, 16, bs, lr, 'gpu10', r'Dense-SGD (lr=%.4f)'%lr, isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True, force_color='r')
    plts.append(line)
    density=0.001
    lr=0.08
    line = plot_with_params(network, 16, bs, lr, 'gpu19', r'LAGS-SGD (lr=%.4f)'%lr,isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
    lr=0.05
    line = plot_with_params(network, 16, bs, lr, 'gpu11', r'LAGS-SGD (lr=%.4f)'%lr,isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='black')
    lr=0.04
    line = plot_with_params(network, 16, bs, lr, 'gpu19', r'LAGS-SGD (lr=%.4f)'%lr,isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='blue')
    lr=0.01
    line = plot_with_params(network, 16, bs, lr, 'gpu10', r'LAGS-SGD (lr=%.4f)'%lr,isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='y')
    ax.legend()


def _get_log_filename(prefix, dnn, nw, bs, lr, sparsity, nsupdate, sg, density, hostname):
     postfix='5922'
     if prefix.find('allreduce')>=0:
         postfix='0'
     if sparsity:
         logfile = '%s/%s/%s-n%d-bs%d-lr%.4f-s%.5f' % (LOGHOME, prefix, dnn, nw, bs, lr, sparsity)
     elif nsupdate:
         logfile = '%s/%s/%s-n%d-bs%d-lr%.4f-ns%d' % (LOGHOME, prefix, dnn, nw, bs, lr, nsupdate)
     else:
         logfile = '%s/%s/%s-n%d-bs%d-lr%.4f' % (LOGHOME, prefix, dnn, nw, bs, lr)
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


def read_speed(logfile):
    f = open(logfile, 'r')
    speeds = []
    computations = []
    forwards = []
    backwards = []
    iotimes = []
    compressions = []
    num_of_layers = 0
    for line in f.readlines():
        if line.find('Speed') > 0:
            speedstr = line.split('Speed: ')[-1].split(' ')[0]
            speed = float(speedstr)
            speeds.append(speed)
        if line.find('average forward and backward time') > 0:
            comptime = line.split('average forward and backward time: ')[-1].split(',')[0]
            comptime = float(comptime)
            computations.append(comptime)
        if line.find(', average forward (') > 0:
            forwardtime = float(line.split(', average forward (')[1].split(')')[0])
            backwardtime = float(line.split('and backward (')[1].split(')')[0])
            #print('line: ', line)
            #print('backwardtime: ', backwardtime)
            iotime = float(line.split('iotime: ')[-1][:-1])
            forwards.append(forwardtime)
            backwards.append(backwardtime)
            iotimes.append(iotime)
        if line.find('Number of groups:') > 0:
            num_of_layers = int(line.split('Number of groups:')[-1][:-1])
        if line.find('Total compress:') > 0:
            #2019-05-20 14:16:07,376 [hv_distributed_optimizer.py:379] INFO [0]: Total compress: 0.009816, allreduce: 0.071333, update: 0.025172, total: 0.106321
            linestr = line.split('Total compress:')[-1]
            compression_time = float(linestr.split(',')[0])
            #update_time = float(linestr.split(',')[-1].split(':')[-1])
            #compressions.append(compression_time+update_time)
            compressions.append(compression_time)
        elif line.find('total[') > 0:
            #2019-05-22 02:00:24,768 [allreducer.py:817] INFO [rank:0]total[269722]: 0.000000,0.000127,0.001726,0.110614,0.003620,0.000161,0.007014
            compression_time = float(line.split(':')[-1].split(',')[2])+float(line.split(':')[-1].split(',')[4])
            compressions.append(compression_time)
    if len(compressions) == 0:
        compressions.append(0)
    si = 1
    avg_speed = np.mean(speeds[si:])
    std_speed = np.std(speeds)
    f.close()
    print('avg backward: ', np.mean(backwards[si:]), ' std: ', np.std(backwards[si:]))
    print('avg speed: ', avg_speed, ' std: ', std_speed)
    return avg_speed, np.mean(iotimes[si:]), np.mean(forwards[si:]), np.mean(backwards[si:]), np.mean(compressions[si:]), num_of_layers

def plotspeed_with_params(dnn, nworkers, bs, lr, hostname, legend, isacc=False, prefix='', title='ResNet-20', sparsity=None, nsupdate=None, sg=None, density=None, force_legend=False):
    nws = []
    avg_speeds = []
    avg_times = []
    avg_computation_times = []
    avg_compression_times = []
    if type(nworkers) is list:
        for nw in nworkers:
            if nw == 1:
                if dnn in ['vgg16', 'resnet20']:
                    logfile = _get_log_filename('singlegpu-baseline-gwarmup-dc1-model-ijcai-wu2norm', dnn, nw, bs, lr, None, 1, None, None, hostname)
                else:
                    logfile = _get_log_filename('singlegpu-baseline-gwarmup-dc1-model-general-1n2w', dnn, nw, bs, lr, None, 1, None, None, hostname)
            else:
                logfile = _get_log_filename(prefix, dnn, nw, bs, lr, sparsity, nsupdate, sg, density, hostname)
            avg_speed, avg_computation, avg_compression = read_speed(logfile)
            avg_speeds.append(avg_speed*nw)
            avg_times.append(bs/avg_speed)
            avg_computation_times.append(avg_computation)
            avg_compression_times.append(avg_compression)
            #avg_speeds.append(avg_speed)
            nws.append(nw)
    else:
        logfile = _get_log_filename(prefix, dnn, nworkers, bs, lr, sparsity, nsupdate)
        avg_speed, avg_computation, avg_computation = read_speed(logfile)
        avg_computation_times.append(avg_computation)
        avg_compression_times.append(avg_compression)
        avg_speeds.append(avg_speed)
        nws.append(nworkers)
    algo = legend
    print('algo: ', algo, ', avgs: ', avg_speeds)
    print('algo: ', algo, ', logfile: ', logfile)
    eff = [avg/avg_speeds[0] for avg in avg_speeds]
    start = 0
    #ax.plot(nws[start:], eff[start:], label=legend, color=fixed_colors[algo])
    opts = [avg_speeds[0]*2**i for i in range(len(avg_speeds))]
    marker=STANDARD_MARKERS[legend.lower()]
    ax.plot(nws, avg_speeds, label=legend, color=fixed_colors[legend.lower()], marker=marker, linewidth=1)
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
    plt.subplots_adjust(bottom=0.18, left=0.19, right=0.98, top=0.9)
    return np.array(avg_speeds), np.array(avg_times), np.array(avg_computation_times), np.array(avg_compression_times)

def plot_breakdowns(ax, comp_times, comm_times, sparse_times, algos):
    #fig, ax = plt.subplots(figsize=(2.8,3.4))
    print('comp_times: ', comp_times)
    print('comm_times: ', comm_times)
    print('sparse_times: ', sparse_times)

    count = len(algos)
    ind = np.arange(count)
    width = 0.4
    margin = 0.05
    xticklabels = algos 
    newind = np.arange(count)
    names = ['Computation', 'Compression+Decompression', 'Communication']
    p1 = ax.bar(newind, comp_times, width, color=Color.comp_color,edgecolor='black',  label=names[0])
    p2 = ax.bar(newind, sparse_times, width, bottom=comp_times, color=Color.compression_color,edgecolor='black', label=names[1])
    p3 = ax.bar(newind, comm_times, width, bottom=comp_times+sparse_times, color=Color.opt_comm_color,edgecolor='black',label=names[2])
    ax.text(10, 10, 'ehhlo', color='b')
    total_times = comp_times+comm_times+sparse_times
    total_times = ['%.3f' % t for t in total_times]
    u.autolabel(p3, ax, total_times, rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    #ax.legend([handles[0][0]], [labels[0][0]], ncol=2)
    print(labels)
    print(handles)
    #ax.set_xlim(left=1+0.3)
    ax.set_ylim(top=ax.get_ylim()[1]*1.15)
    ax.set_xticks(ind)
    ax.set_xticklabels(xticklabels)
    #ax.set_xlabel('Model')
    ax.set_ylabel('time [sec]')
    u.update_fontsize(ax, 10)
    #ax.legend((p1[0], p2[0], p3[0]), tuple(names), ncol=9, bbox_to_anchor=(1, -0.1))#, handletextpad=0.2, columnspacing =1.)


def plot_time_breakdowns_bar():
    global ax
    global FONTSIZE
    global LOGHOME
    LOGHOME='/home/shshi/work/p2p-dl/logs'
    dnn='resnet50';density=0.001;bs=16;lr=0.01;nw=16
    dense_logfile = _get_log_filename('allreduce-gwarmup-dc1-model-thres-8kbytes-infocom-debug', dnn=dnn, nw=nw, bs=bs, lr=lr, sparsity=None, nsupdate=1, sg=None, density=1.0, hostname='MGD')
    avg_speed, avg_io, avg_fw, avg_bw, avg_sparse = read_speed(dense_logfile)
    avg_time = bs/avg_speed

    avg_comm = avg_time - avg_io - avg_fw - avg_bw - avg_sparse
    print('[Dense] io, forward, backward, sparse, communication, total')
    print(dnn, nw, avg_io, avg_fw, avg_bw, avg_sparse, avg_comm, avg_time)

def read_single_node():
    global LOGHOME
    LOGHOME='/home/shshi/work/p2p-dl/logs';
    #dnn='resnet50';bs=16;
    #dnn='googlenet';bs=32;
    #dnn='vgg16i';bs=32;
    #dnn='inceptionv4';bs=32;
    dnn='alexnet';bs=256;
    fn = '%s/singlegpu-gwarmup-dc1-model-infocom-thestest/%s-n1-bs%d-lr0.0100-ns1/MGD.log' % (LOGHOME, dnn, bs)
    print('fn: ', fn)
    avg_speed, avg_io, avg_fw, avg_bw, avg_sparse, num_of_layers = read_speed(fn)
    avg_time = bs/avg_speed
    print(dnn, 1, 0, num_of_layers, avg_io, avg_fw, avg_bw, 0, 0, avg_time)

def read_multiple_speeds():
    global LOGHOME
    #LOGHOME='/home/shshi/work/p2p-dl/logs';host='MGD'
    LOGHOME='./logs';host='scigpu10'; #host='hsw224'

    # NVIDIA
    #dnn='resnet50';density=0.001;bs=256;lr=0.01;nw=16
    dnn='resnet50';density=0.001;bs=128;lr=0.01;nw=16
    dnn='vgg16i';density=0.001;bs=128;lr=0.01;nw=16
    dnn='inceptionv4';density=0.001;bs=128;lr=0.01;nw=16
    dnn='alexnet';density=0.001;bs=128;lr=0.01;nw=16
    thresholds= [ 524288000][::-1];ada=False
    print('[Dense] io, forward, backward, sparse, communication, total')
    speeds = []
    for thres in thresholds:
        if ada:
            density=1;prefix='allreduce-comp-topk-gwarmup-dc1-model-infocom20-final-ada-thres-0kbytes'
        else:
            #density=1;prefix='allreduce-comp-topk-gwarmup-dc1-model-tpds-prod-thres-%dkbytes' % (thres/1024)
            density=0.001;prefix='allreduce-comp-redsynctrim-gwarmup-dc1-model-iclr-rebuttal-speed-thres-512000kbytes'
        dense_logfile = _get_log_filename(prefix, dnn=dnn, nw=nw, bs=bs, lr=lr, sparsity=None, nsupdate=1, sg=None, density=density, hostname=host)
        avg_speed, avg_io, avg_fw, avg_bw, avg_sparse, num_of_layers = read_speed(dense_logfile)
        avg_time = bs/avg_speed
        avg_comm = avg_time - avg_io - avg_fw - avg_bw - avg_sparse
        speeds.append((dnn, nw, thres, num_of_layers, avg_io, avg_fw, avg_bw, avg_sparse, avg_comm, avg_time))
    for s in speeds:
        print(*s)

def plot_time_breakdowns():
    global ax
    global FONTSIZE
    global LOGHOME
    LOGHOME='/home/shshi/work/p2p-dl/logs'
    FONTSIZE = 12
    #networks = ['vgg16', 'resnet20', 'lstm', 'lstman4']
    networks = ['resnet20', 'vgg16', 'lstm']#, 'lstm']#, 'lstman4']
    #networks = ['resnet20', 'vgg16', 'inceptionv4']#, 'lstm']#, 'lstman4']
    #networks = ['inceptionv4', 'vgg16']#, 'lstm']#, 'lstman4']
    #networks = ['vgg16']#, 'lstm']#, 'lstman4']
    networks = ['googlenet', 'resnet50']
    num_rows = NFIGURES/NFPERROW
    for i, network in enumerate(networks):
        ax_row = i / NFPERROW
        ax_col = i % NFPERROW
        if NFIGURES/NFPERROW == 1:
            ax = group_axs[i] if len(group_axs) > 0 else group_axs
        else:
            ax = group_axs[ax_row][ax_col]
        plts = plot_breakdowns_with_network(ax, network)
        ax.grid(linestyle=':')
        ax.set_title(STANDARD_TITLES[network])
        u.update_fontsize(ax, FONTSIZE)
    lines, labels = ax.get_legend_handles_labels()

    if num_rows == 1:
        fig.legend(lines, labels, ncol=4, loc='upper center', fontsize=12, frameon=True)
        plt.subplots_adjust(bottom=0.1, left=0.07, right=0.99, top=0.78, wspace=0.32, hspace=0.49)
    else:
        fig.legend(lines, labels, ncol=4, loc='upper center', fontsize=FONTSIZE, frameon=True)
        plt.subplots_adjust(bottom=0.08, left=0.09, right=0.97, top=0.88, wspace=0.27, hspace=0.49)
    plt.savefig('%s/group_timebreakdown.pdf'%OUTPUTPATH)

def plot_breakdowns_with_network(ax, network):
    dnn=network
    if dnn=='resnet20':
        dnn='resnet20';density=0.001;bs=32;lr=0.1;nw=16
    elif dnn=='vgg16':
        dnn='vgg16';density=0.001;bs=32;lr=0.1;nw=16
    elif dnn=='lstm':
        dnn='lstm';density=0.001;bs=20;lr=22.0;nw=16
    elif dnn=='alexnet':
        density=0.01;bs=64;lr=0.01;nw=16
    elif dnn=='inceptionv4':
        density=0.004;bs=32;lr=0.01;nw=16
    else:
        density=0.01;bs=32;lr=0.01;nw=16
    dense_logfile = _get_log_filename('allreduce-baseline-gwarmup-dc1-model-infocom-debug', dnn=dnn, nw=nw, bs=bs, lr=lr, sparsity=None, nsupdate=1, sg=None, density=1.0, hostname='MGD')
    topk_logfile = _get_log_filename('allreduce-comp-sigmathresallgather-baseline-gwarmup-dc1-model-nips-layerwise-speedfinal-adaptive', dnn=dnn, nw=nw, bs=bs, lr=lr, sparsity=None, nsupdate=1, sg=2.5, density=density, hostname='MGD')
    lw_logfile = _get_log_filename('allreduce-comp-sigmathresallgather-baseline-gwarmup-dc1-model-nips-layerwise-speedfinal-adaptive', dnn=dnn, nw=nw, bs=bs, lr=lr, sparsity=None, nsupdate=1, sg=None, density=density, hostname='MGD')
    group_computation = []
    group_communication = []
    group_sparsification = []
    avg_speed, avg_computation, avg_sparse = read_speed(dense_logfile)
    avg_speed=bs/avg_speed
    print('-----DNN:%s dense speed: ' % dnn, avg_speed)
    avg_comm = avg_speed - avg_computation
    dense_computation = avg_computation # used as reference
    group_computation.append(avg_computation)
    group_communication.append(avg_comm)
    #group_computation.append(0)
    #group_communication.append(0)
    group_sparsification.append(0) # dense version should be zero

    topk_speed, topk_computation, topk_sparse = read_speed(topk_logfile)
    topk_speed=bs/topk_speed
    print('------DNN:%s topk speed: '%dnn, topk_speed)
    topk_computation = dense_computation
    topk_comm = topk_speed - topk_computation - topk_sparse
    group_computation.append(topk_computation)
    group_communication.append(topk_comm)
    group_sparsification.append(topk_sparse) 

    lw_speed, lw_computation, lw_sparse = read_speed(lw_logfile)
    lw_speed=bs/lw_speed
    print('------DNN:%s lw speed: ' % dnn, lw_speed)
    lw_sparse = lw_computation - dense_computation
    lw_comm = lw_speed - lw_computation  
    lw_computation = dense_computation
    #lw_computation -= lw_sparse
    group_computation.append(lw_computation)
    group_communication.append(lw_comm)
    group_sparsification.append(lw_sparse) 
    algos=['Dense-SGD', 'SLGS-SGD', 'LAGS-SGD']
#def plot_breakdowns(ax, comp_times, comm_times, sparse_times, algos):
    plot_breakdowns(ax, np.array(group_computation), np.array(group_communication), np.array(group_sparsification), algos)


def plot_throughputs():
    global LOGHOME
    global FONTSIZE
    FONTSIZE=12
    LOGHOME='/home/shshi/work/p2p-dl/logs'
    dnn='resnet20';density=0.001;bs=32;lr=0.1;nworkers=[2, 4, 8, 16] #, 8, 16, 32, 64]
    dense_speeds, dense_times, dense_computation_times, dense_compression_times = plotspeed_with_params(dnn=dnn, nworkers=nworkers, bs=bs, lr=lr, hostname='MGD', legend=r'Dense-SGD', prefix='allreduce-baseline-gwarmup-dc1-model-nips-layerwise-speed2-adaptive', title=dnn, sparsity=None, nsupdate=1, density=1.0, force_legend=True)
    topk_speeds, topk_times, topk_computation_times, topk_compression_times = plotspeed_with_params(dnn=dnn, nworkers=nworkers, bs=bs, lr=lr, hostname='MGD', legend=r'SLGS-SGD', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speed2-adaptive', title=dnn, sparsity=None, nsupdate=1, sg=2.5, density=density, force_legend=True)
    layerwise_topk_speeds, layerwise_topk_times, lw_computation_times, lw_compression_times = plotspeed_with_params(dnn=dnn, nworkers=nworkers, bs=bs, lr=lr, hostname='MGD', legend=r'LAGS-SGD', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speed2-adaptive', title=dnn, sparsity=None, nsupdate=1, density=density, force_legend=True)
    topk_speedups = layerwise_topk_speeds/topk_speeds
    dense_speedups = layerwise_topk_speeds/dense_speeds
    print(dnn, ', topk_speedups: ', topk_speedups[-1])
    print(dnn, ', dense_speedups: ', dense_speedups[-1])
    print(dnn, ' dense times: ', dense_times, ', computations: ', dense_computation_times, ', compression: ', dense_compression_times)
    print(dnn, ' topk times: ', topk_times, ', computations: ', topk_computation_times, ', compression: ', topk_compression_times)
    print(dnn, ' layerwise topk times: ', layerwise_topk_times, ', computations: ', lw_computation_times, ', compression: ', lw_compression_times)
    ax.grid(linestyle=':')
    plt.legend(fontsize=10, loc='upper left')
    plt.savefig('%s/%s_throughput.pdf' % (OUTPUTPATH, dnn))

def plot_gradient_hist():
    #compressor='topk'
    compressor='gaussion'
    dnn='resnet20'
    #LOGHOME='/datasets/shshi/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-thres-512000kbytes/%s-n8-bs32-lr0.1000-ns1-ds0.001' % (compressor, dnn);
    #LOGHOME='./logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-debug-thres-512000kbytes/%s-n8-bs32-lr0.1000-ns1-ds0.001' % (compressor, dnn);
    LOGHOME='./logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-logging-thres-512000kbytes/%s-n16-bs32-lr0.1000-ns1-ds0.001' % (compressor, dnn);
    iterations=range(800, 1700) #[1000, 1100, 1200, 1900]
    for i in iterations:
        if i % 200 != 0:
            continue
        #fn = '%s/gradients/%s/%s/r0_gradients_iter_%d.npy' % (LOGHOME, type, dnn, i)
        fn = '%s/r0_gradients_iter_%d.npy' % (LOGHOME, i)
        grad = np.load(fn)
        d = grad.size
        k = int(0.001*d)
        sorted = np.abs(grad).argsort()
        gradcopy = grad.copy()
        indexes = sorted[-k:]
        non_indexes = sorted[0:d-k]
        gradcopy[non_indexes] = 0.0
        topknorm = np.power(np.linalg.norm(grad-gradcopy), 2)/np.power(np.linalg.norm(grad), 2)
        print('Iter:', i, ': ||x-topk(x)||^2/||x||^2 = ', topknorm)
        print('Iter:', i, ': 1-k/d = ', 1-k*1.0/d)
        print('Iter:', i, ': 0.01||x||^2 = ', 0.01*np.linalg.norm(grad))
        u.plot_hist(grad, title='iter-%d'%i, ax=ax)

    #plt.savefig('%s/%s_grads_iter%d.pdf' % (OUTPUTPATH, dnn, iteration))
    plt.legend()
    plt.show()

def timebreaks():
    fig, ax = plt.subplots(figsize=(6.2,4.4))
    xticklabels = ['AlexNet', 'VGG-16', 'ResNet-50', 'Inception-V4']
    dnns = ['alexnet', 'vgg16', 'resnet50', 'inceptionv4']
    algos = ['dense-sgd', 'topk-sgd', 'dgc-sgd', 'gaussian-sgd']
    data = {'alexnet':  # [forward+backward, sparse, communication]
                    {'topk-sgd': [0.08, 0.0, 0.0767],
                     'omgs-sgd': [0.021, 0.0313, 0.0737, 0.205],
                     }, 
            'resnet50':
                    {'topk-sgd': [0.09, 0.12, 0.1376, 0.3185],
                      'omgs-sgd': [0.09, 0.12, 0.1321, 0.165],
                     }, 
            'inceptionv4': 
                    {'topk-sgd': [0.305, 0.524, 0.2368, 0.5357],
                      'omgs-sgd': [0.305, 0.524, 0.22, 0.19409],
                     }, 
            'lstm':
                    {'topk-sgd': [0.076, 0.13089, 0.37548, 0.437],
                      'omgs-sgd': [0.076, 0.13089, 0.33875, 0.3705],
                     }, 
            }
    count = len(dnns)
    width = 0.4; margin = 0.07
    s = (1 - (width*count+(count-1) *margin))/2+width
    ind = np.array([s+i+1 for i in range(count)])
    labels=['TopK-S.', 'OMGS-S.']
    
    for i, algo in enumerate(algos):
        newind = ind+s*width+(s+1)*margin
        forward = []; backward=[];sparse=[];commu=[]
        for dnn in dnns:
            d = data[dnn]
            ald = d[algo]
            forward.append(ald[0])
            backward.append(ald[1])
            sparse.append(ald[2])
            commu.append(ald[3])
        p1 = ax.bar(newind, forward, width, color=Color.forward_color,edgecolor='black', label='Forward')
        p2 = ax.bar(newind, backward, width, bottom=np.array(forward), color=Color.backward_color,edgecolor='black', label='Backward')
        p3 = ax.bar(newind, sparse, width, bottom=np.array(forward)+np.array(backward), color=Color.compression_color,edgecolor='black', label='Sparsification')
        p4 = ax.bar(newind, commu, width, bottom=np.array(forward)+np.array(backward)+np.array(sparse), color=Color.comm_color,edgecolor='black', label='Communication')
        s += 1 
        #ax.text(4, 4, 'ehhlo', color='b')
        u.autolabel(p4, ax, labels[i], 0, 8)
    ax.set_ylim(top=ax.get_ylim()[1]*1.05)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend((p1[0], p2[0], p3[0], p4[0]), (labels[0],labels[1], labels[2], labels[3] ), ncol=1, handletextpad=0.2, columnspacing =1., loc='upper left')
    ax.set_ylabel('Time [s]')
    ax.set_xticks(newind-width/2-margin/2)
    ax.set_xticklabels(xticklabels)
    plt.savefig('%s/timebreakdown.pdf' % (OUTPUTPATH), bbox_inches='tight')

def verify_t_distribution():
    size = 10000
    d = np.random.standard_t(4, size=size)
    from statsmodels.graphics.gofplots import qqplot
    qqplot(d, line='s')
    plt.show()

def compare_topk_and_gaussiank():
    topkfn='./logs/topk-largesize.log'
    topksizes, topktimes = readers.read_topkperf_log(topkfn)

    gfn='./logs/gaussion-largesize.log'
    gsizes, gtimes = readers.read_topkperf_log(gfn)

    gfn='./logs/dgcsampling-largesize.log'
    dsizes, dtimes = readers.read_topkperf_log(gfn)

    ax.plot(topksizes, topktimes, label='Top-k', color='r')
    ax.plot(dsizes, dtimes, label='DGC-k', color='b')
    ax.plot(gsizes, gtimes, label='Gaussian-k', color='g')
    #ax.set_xscale('log')
    #ax.set_xlabel(r'log($d$)')
    #ax.set_xlabel('# of parameters in logarithmic scale')
    ax.set_xlabel('# of parameters ')
    ax.set_ylabel('GPU computation time (s)')
    ax.grid(linestyle=':')
    FONTSIZE=12
    ax.legend(fontsize=FONTSIZE)
    u.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/topkvsgaussiank.pdf' % (OUTPUTPATH), bbox_inches='tight')
    plt.show()

def plot_communication_vs_accuracy():
    global ax
    global FONTSIZE
    global LOGHOME
    global global_max_epochs
    LOGHOME='./logs'
    FONTSIZE=14
    isacc = True
    labels = {
            'gaussion': 'GaussianK-SGD',
            'dgcsampling': 'DGC-SGD',
            'redsynctrim': 'RedSync-SGD',
            'topk': 'TopK-SGD'
            }

    def _read_comm_vs_accs(logfile):
        print('logfile: ', logfile)
        f = open(logfile)
        comms = []
        sum_comms = []
        accs = []
        for l in f.readlines():
            if l.find('val top-1') > 0:
                acc = float(l.split('val top-1 acc:')[1].split(',')[0])
                accs.append(acc)
            if l.find('Average number of selected gradients:') > 0:
                comm = float(l.split('Average number of selected gradients:')[1].split(',')[0])
                comms.append(comm)
        sum_comms = comms[:]
        for i in range(1, len(comms)): 
            sum_comms[i] += sum_comms[i-1]+comms[i]
        sum_comms = sum_comms[0:global_max_epochs]
        accs = accs[0:global_max_epochs]
        #sum_comms = range(global_max_epochs)
        return np.array(sum_comms), np.array(accs)

    #dnn='vgg16';global_max_epochs = 120;bs=128;lr=0.1;nworkers=16;num_iters_per_epoch=25
    dnn='resnet20';global_max_epochs = 120;bs=32;lr=0.1;nworkers=16;num_iters_per_epoch=98
    density=0.001;compressor='gaussion'
    prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-overorunderspar-thres-512000kbytes' % compressor
    logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-ns1-ds%s' % (prefix, dnn, nworkers, bs, lr, str(density)) +  '/scigpu10-0.log'
    sum_comms, accs = _read_comm_vs_accs(logfile)
    sum_comms *= num_iters_per_epoch
    line = ax.plot(sum_comms, accs, label=labels[compressor], marker=None, markerfacecolor='none', color=colors[1], linewidth=1)

    #density=0.001;compressor='redsynctrim'
    #prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-overorunderspar-thres-512000kbytes' % compressor
    #logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-ns1-ds%s' % (prefix, dnn, nworkers, bs, lr, str(density)) +  '/scigpu10-0.log'
    #sum_comms, accs = _read_comm_vs_accs(logfile)
    #line = ax.plot(sum_comms, accs, label=labels[compressor], marker=None, markerfacecolor='none', color=colors[0], linewidth=1)

    density=0.001;compressor='dgcsampling'
    prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-overorunderspar-thres-512000kbytes' % compressor
    logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-ns1-ds%s' % (prefix, dnn, nworkers, bs, lr, str(density)) +  '/scigpu10-0.log'
    sum_comms, accs = _read_comm_vs_accs(logfile)
    sum_comms *= num_iters_per_epoch
    line = ax.plot(sum_comms, accs, label=labels[compressor], marker=None, markerfacecolor='none', color=colors[2], linewidth=1)

    density=0.001;compressor='topk'
    prefix='allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-overorunderspar-thres-512000kbytes' % compressor
    logfile = LOGHOME+'/%s/%s-n%d-bs%d-lr%.4f-ns1-ds%s' % (prefix, dnn, nworkers, bs, lr, str(density)) +  '/scigpu10-0.log'
    sum_comms, accs = _read_comm_vs_accs(logfile)
    sum_comms *= num_iters_per_epoch
    line = ax.plot(sum_comms, accs, label=labels[compressor], marker=None, markerfacecolor='none', color=colors[3], linewidth=1)


    ax.set_xlabel('number of communicated gradients')
    ax.set_ylabel('val accuracy')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    #ax.set_xscale('log')
    ax.grid(linestyle=':')
    ax.legend()
    plt.savefig('%s/commvsacc_%s.pdf' % (OUTPUTPATH, dnn), bbox_inches='tight')



if __name__ == '__main__':
    if PLOT_TYPE == 'NORM':
        plot_group_norm_diff()
    elif PLOT_TYPE == 'ACC':
        plot_group_convergences()
        #verify_t_distribution()
    elif PLOT_TYPE == 'THROUGHPUT':
        #plot_throughputs()
        #plot_time_breakdowns()
        plot_time_breakdowns_bar()
    else:
        #plot_others()
        #read_multiple_speeds()
        #timebreaks()
        #read_single_node()
        #plot_gradient_hist()
        #plot_single_convergence()
        #plot_single_convergence_sensitivity()
        plot_communication_vs_accuracy()
        #compare_topk_and_gaussiank()

    plt.show()

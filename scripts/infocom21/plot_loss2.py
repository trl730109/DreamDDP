# -*- coding: utf-8 -*-
from __future__ import print_function
import time
from matplotlib import rcParams
FONT_FAMILY='DejaVu Serif'
rcParams["font.family"] = FONT_FAMILY 
from mpl_toolkits.axes_grid.inset_locator import inset_axes, zoomed_inset_axes
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from plot_sth import Bar
import plot_sth as Color
#rc('text', usetex=True)
#matplotlib.use("TkAgg")
import numpy as np
import datetime
import itertools
import utils as u
PLOT_TYPE = 'NORM'
PLOT_TYPE = 'ACC'
#PLOT_TYPE = 'THROUGHPUT'
#PLOT_TYPE = 'OTHERS'
PLOT_TYPE = 'NONE'

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


OUTPUTPATH='/media/sf_Shared_Data/tmp/infocom20'
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
    FIGSIZE=(2*4.5*NFPERROW,2*3.2)

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
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG-16',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50',
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
        valid = line.find('val acc: ') > 0 if isacc else line.find('avg loss: ') > 0
        #valid = line.find('avg train acc: ') > 0 if isacc else line.find('avg loss: ') > 0
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
        if line.find('num_batches_per_epoch: ') > 0:
            num_batches_per_epoch = int(line[0:-1].split('num_batches_per_epoch:')[-1])
        valid = line.find('val acc: ') > 0 if isacc else line.find('average loss: ') > 0
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
        print('max validation accuracy: ', np.max(losses))
        if logfile.find('lstm') > 0:
            print('min ppl: ', np.min(losses))


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
        ax.set_ylabel('val accuracy')
        if logfile.find('lstm') > 0:
            ax.set_ylabel('perplexity')
            
    else:
        ax.set_ylabel('training loss')
    #plt.title('ResNet-50')
    ax.set_title(get_real_title(title))
    #marker = fixed_markers.get(label.lower(), None) #markeriter.next()
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
        ax.set_xlabel('# of epochs')
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
    LOGHOME='/media/sf_Shared_Data/gpuhome/repositories/p2p-dl/logs'
    FONTSIZE = 14
    #networks = ['vgg16', 'resnet20', 'lstm', 'lstman4']
    #networks = ['resnet20', 'vgg16', 'lstm']#, 'lstm']#, 'lstman4']
    #networks = ['resnet20', 'vgg16', 'resnet50']#, 'lstm']#, 'lstman4']
    #networks = ['alexnet', 'vgg16', 'resnet50']#, 'lstm']#, 'lstman4']
    networks = ['vgg16', 'resnet50', 'lstm']#, 'lstman4']
    #networks = ['vgg16', 'resnet20', 'alexnet', 'resnet50', 'lstm', 'lstman4']
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
        #plot_sub_boxes(ax, network)
    lines, labels = ax.get_legend_handles_labels()

    if num_rows == 1:
        fig.legend(lines, labels, ncol=4, loc='upper center', fontsize=12, frameon=True)
        plt.subplots_adjust(bottom=0.17, left=0.07, right=0.99, top=0.80, wspace=0.32, hspace=0.49)
    else:
        fig.legend(lines, labels, ncol=4, loc='upper center', fontsize=FONTSIZE, frameon=True)
        plt.subplots_adjust(bottom=0.08, left=0.09, right=0.97, top=0.88, wspace=0.27, hspace=0.49)
    plt.savefig('%s/group_convergence.pdf'%OUTPUTPATH)

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
        #line = plot_with_params(network, 16, 20, lr, 'gpu10', r'SLGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-convtune-adaptive-qiang', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
        #plts.append(line)
        density=0.004
        line = plot_with_params(network, 16, 20, lr, 'MGD', r'OMGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-convtune-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        plts.append(line)
    elif network == 'resnet20':
        global_max_epochs = 120;bs=32;lr=0.1
        density=1.0
        line = plot_with_params(network, 16, 32, lr, 'gpu19', r'Dense-SGD', isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-model-nips-layerwise-conv2-adaptive', nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        density=0.001
        line = plot_with_params(network, 16, 32, lr, 'gpu19', r'SLGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-conv2-adaptive', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
        plts.append(line)
        density=0.001
        line = plot_with_params(network, 16, 32, lr, 'gpu19', r'LAGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-conv2-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        plts.append(line)
    elif network == 'vgg16':
        global_max_epochs = 120;bs=32;lr=0.1;nworkers=32
        density=1.0
        line = plot_with_params(network, 16, 32, lr, 'gpu16', r'P-SGD', isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-model-nips-layerwise-conv2-adaptive', nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        density=0.001
        #line = plot_with_params(network, 16, 32, lr, 'gpu16', r'SLGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-conv2-adaptive', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
        #plts.append(line)
        density=0.001
        line = plot_with_params(network, 16, 32, lr, 'gpu19', r'OMGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-conv2-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        plts.append(line)
    elif network == 'lstman4':
        global_max_epochs = 120;bs=32;lr=0.0002;nworkers=32
        density=1.0
        line = plot_with_params(network, 16, 4, lr, 'gpu19', r'P-SGD', isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-model-nips-layerwise-convlstm-adaptive', nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        density=0.001
        line = plot_with_params(network, 16, 4, lr, 'gpu19', r'SLGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-convlstm-adaptive', nsupdate=1, sg=2.5, density=density, force_legend=True,force_color='blue')
        plts.append(line)
        density=0.001
        line = plot_with_params(network, 16, 4, lr, 'gpu19', r'LAGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-convlstm-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        plts.append(line)
    elif network == 'resnet50':
        global_max_epochs = 120;bs=32;lr=0.01
        density=1.0
        line = plot_with_params(network, 16, 32, lr, 'gpu10', r'P-SGD', isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        density=0.001
        line = plot_with_params(network, 16, 32, lr, 'gpu10', r'OMGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        plts.append(line)
    elif network == 'alexnet':
        global_max_epochs = 120;bs=128;lr=0.01
        density=1.0
        line = plot_with_params(network, 16, bs, lr, 'gpu10', r'Dense-SGD', isacc=isacc, prefix='allreduce-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True, force_color='r')
        plts.append(line)
        density=0.001
        lr=0.08
        line = plot_with_params(network, 16, bs, lr, 'gpu19', r'LAGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        lr=0.05
        line = plot_with_params(network, 16, bs, lr, 'gpu11', r'LAGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        lr=0.04
        line = plot_with_params(network, 16, bs, lr, 'gpu19', r'LAGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        lr=0.01
        line = plot_with_params(network, 16, bs, lr, 'gpu10', r'LAGS-SGD',isacc=isacc,  prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-nips-layerwise-speedk80-adaptive', nsupdate=1, density=density, force_legend=True,force_color='g')
        plts.append(line)
    return plts


def _get_log_filename(prefix, dnn, nw, bs, lr, sparsity, nsupdate, sg, density, hostname, nstreams=None):
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
     if nstreams is not None:
         logfile += '-nstreams%d' % nstreams 
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
    LOGHOME='./logs';host='gpu9'; #host='hsw224'
    # CJ 
    #logprefix='nstreams-rdma-hvd'
    #logprefix='nstreams-infocom2021-ada'
    #logprefix='nstreams-infocom2021-eth-asc-ada'
    #interface='100GbE'
    interface='eth'
    #interface='rdma'
    ASC=True
    #ASC=False
    if interface == 'rdma':
        if ASC:
            logprefix='nstreams-infocom2021-asc-ada'
        else:
            logprefix='nstreams-infocom2021-ada'
    else:
        if ASC:
            logprefix='nstreams-infocom2021-%s-asc-ada' % interface
        else:
            logprefix='nstreams-infocom2021-%s-ada' % interface
    dnn='resnet50';density=1;bs=64;lr=1.2;nw=32
    dnn='densenet201';density=1;bs=32;lr=1.2;nw=32
    dnn='densenet161';density=1;bs=32;lr=1.2;nw=32
    dnn='resnet152';density=1;bs=32;lr=1.2;nw=32
    #dnn='vgg16i';density=1;bs=64;lr=0.01;nw=32
    #dnn='googlenet';density=0.001;bs=64;lr=0.01;nw=16
    dnn='inceptionv4';density=1;bs=32;lr=1.2;nw=32
    #dnn='alexnet';density=0.001;bs=64;lr=0.01;nw=16
    MB = 1024*1024
    #thresholds= [ 0.5, 1, 2, 8, 16, 32, 64, 128, 256]
    thresholds= [ 500][::-1]
    #thresholds= [ 524288][::-1]
    thresholds = [int(t*MB) for t in thresholds]
    #tensorfusions=[True]
    tensorfusions=[False]
    nstreams_list = list(range(1, 5))
    print('[Dense] io, forward, backward, sparse, communication, total')
    speeds = []
    for thres in thresholds:
        for tensorfusion in tensorfusions:
            #density=1.0;prefix='allreduce-gwarmup-dc1-model-%s-thres-%dkbytes'%(logprefix, thres//1024)
            density=1.0;prefix='allreduce-gwarmup-dc1-model-%s-thres-%dkbytes'%(logprefix, thres//1024)
            #density=1.0;prefix='allreduce-gwarmup-dc1-model-nstreams-chvd-thres-%dkbytes'%(thres//1024)
            #density=1.0;prefix='allreduce-gwarmup-dc1-model-nstreams-ada-thres-%dkbytes'%(thres//1024)
            #density=1.0;prefix='allreduce-gwarmup-dc1-model-nstreams-thres-%dkbytes'%(thres//1024)
            nl = nstreams_list
            if tensorfusion:
                #nl = [1]
                prefix='allreduce-gwarmup-dc1-model-%s-ntf-thres-%dkbytes'%(logprefix, thres//1024)
            for nstreams in nl:
                dense_logfile = _get_log_filename(prefix, dnn=dnn, nw=nw, bs=bs, lr=lr, sparsity=None, nsupdate=1, sg=None, density=density, hostname=host, nstreams=nstreams)
                avg_speed, avg_io, avg_fw, avg_bw, avg_sparse, num_of_layers = read_speed(dense_logfile)
                avg_time = bs/avg_speed
                avg_comm = avg_time - avg_io - avg_fw - avg_bw - avg_sparse
                #speeds.append((dnn, nstreams, nw, thres, num_of_layers, avg_io, avg_fw, avg_bw, avg_sparse, avg_comm, avg_time))
                speeds.append((dnn, nstreams, nw, thres, avg_time, tensorfusion))
    for s in speeds:
        print(*s)
    #s = [s[4] for s in speeds[:-1]]
    #plt.plot(nstreams_list, s, label='w/o tensor fusion')
    #plt.scatter(1, speeds[-1][4], label='tensor fusion', color='red')
    #plt.xlabel('# NCCL streams')
    #plt.ylabel('Iteration time [s]')
    #plt.legend()
    #plt.title(dnn)
    #plt.show()


if __name__ == '__main__':
    read_multiple_speeds()
    plt.show()

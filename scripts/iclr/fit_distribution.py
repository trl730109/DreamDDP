# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes, zoomed_inset_axes
import utils as u

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')
#colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
colors = ['#F1948A', '#C0392B', '#9B59B6', '#2980B9', '#3498DB', '#17A589', '#229954', '#D4AC0D', '#D4AC0D', '#D68910', '#A6ACAF', '#2E4053', '#5B33FF', '#E72913']

OUTPUTPATH='/home/shshi/gpuhome/plots/posticlr'

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    #DISTRIBUTIONS = [        
    #    st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
    #    st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #    st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #    st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #    st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    #    st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
    #    st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #    st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #    st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #    st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    #]
    DISTRIBUTIONS = [        
        st.norm,st.t,st.johnsonsb,st.johnsonsu,st.levy_stable
       ] 

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax, label=distribution.name)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


def plot_sub_boxes(ax, X, Y):

    #Plot subboxes
    bbox_to_anchor = (0.0, 0.2, 1, 0.7)
    subaxes = inset_axes(ax,
        width='30%', 
        height='40%', 
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        loc='upper right')
    idx = 0
    for x, y in zip(X, Y):
        half = len(x)*2//8
        halfend = half+len(x)/4
        subx = x[half:halfend]
        suby = y[half:halfend]
        subaxes.plot(subx, suby, color=colors[idx], linewidth=1.5)
        idx+=1
    #subaxes.set_ylim(bottom=subaxes.get_ylim()[0])

def fit_distribution():
    #compressor='topk'
    #ax, plt.figure(figsize=(5.5,4.0))
    fig, ax= plt.subplots(1, 1,figsize=(5.5,4.0))
    PLOT_THRES=False
    PLOT_CDF = False

    compressor='topk'
    #compressor='none'
    #dnn='resnet20';bs=32;lr=0.1;xlimit=0.023 if compressor=='none' else 0.14
    #dnn='vgg16';bs=128;lr=0.1;xlimit=0.0025 if compressor=='none' else 0.015
    #dnn='fcn5net';bs=128;lr=0.01;xlimit=0.01 if compressor=='none' else 0.1
    dnn='lenet';bs=128;lr=0.01;xlimit=0.04 if compressor=='none' else 0.16
    #dnn='lstm';bs=20;lr=22;xlimit=0.00022 if compressor=='none' else 0.002
    #dnn='lstman4';bs=4;lr=0.0002;xlimit=0.38 if compressor=='none' else 3.5
    #LOGHOME='/datasets/shshi/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-thres-512000kbytes/%s-n8-bs32-lr0.1000-ns1-ds0.001' % (compressor, dnn);
    if compressor == 'none':
        LOGHOME='./logs/iclr/gradients/allreduce-gwarmup-dc1-model-iclr-debug-thres-512000kbytes/%s-n8-bs%d-lr%.4f-ns1-ds1.0' % (dnn, bs, lr);
    else:
        LOGHOME='../p2p-iclr/logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-logging-thres-512000kbytes/%s-n8-bs%d-lr%.4f-ns1-ds0.001' % (compressor, dnn, bs, lr);
        #LOGHOME='./logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-debug-thres-512000kbytes/%s-n8-bs%d-lr%.4f-ns1-ds0.001' % (compressor, dnn, bs, lr);
    iterations=range(800, 1000) #[1000, 1100, 1200, 1900]
    idx=0
    X = []
    Y = []
    for i in iterations:
        if i % 200 != 0:
            continue
        #fn = '%s/gradients/%s/%s/r0_gradients_iter_%d.npy' % (LOGHOME, type, dnn, i)
        fn = '%s/r0_gradients_iter_%d.npy' % (LOGHOME, i)
        grad = np.load(fn).flatten()
        d = len(grad)
        k = int(0.001*d)
        print('# of params: ', d)
        #l2norm = np.linalg.norm(grad)
        #grad /= l2norm
        #data = pd.Series(grad)
        #ax = data.plot(kind='hist', bins=500, normed=True, alpha=0.5, label='iter-%d'%i)

        if PLOT_CDF:
            #CDF
            count, bins, _ = ax.hist(grad, 5000, linewidth=1.5, histtype='step', cumulative=True, normed=True, alpha=1.0, label='iter-%d'%i, color=colors[idx])
        else:
            # Histo
            count, bins, _ = ax.hist(grad, 5000, normed=False, alpha=1.0, label='iter-%d'%i, color=colors[idx])
            #count, bins, _ = ax.hist(grad, 5000, normed=True, alpha=1.0, label='iter-%d'%i, color=colors[idx])
        X.append(count)
        Y.append(bins)
        if PLOT_THRES:
            abs_grad = np.abs(grad)
            sorted_grad = np.sort(abs_grad)[::-1]
            thres = sorted_grad[k]
            y = range(0, int(np.mean(count)*2))
            x = len(y)*[thres]
            ax.plot(x,y, color=colors[idx])
        idx+=1

        #best_fit_name, best_fit_params = best_fit_distribution(grad, 200, ax)
        #ax.legend()
        #best_dist = getattr(st, best_fit_name)
        #pdf = make_pdf(best_dist, best_fit_params)
        ##plt.figure(figsize=(12,8))
        #ax = pdf.plot(lw=2, label=best_fit_name, legend=True)
        #data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

    ax.grid(linestyle=':')
    ax.set_xlabel('gradient value')
    FONTSIZE=14
    if PLOT_CDF:
        ax.legend(fontsize=FONTSIZE, loc=2)
    else:
        ax.legend(fontsize=FONTSIZE, loc=1)
    #plot_sub_boxes(ax, X, Y)
    u.update_fontsize(ax, FONTSIZE)
    #plt.savefig('%s/gradident_distrition_%s.pdf' % (OUTPUTPATH, dnn), bbox_inches='tight')
    if PLOT_CDF:
        ax.set_xlim(left=-xlimit, right=xlimit)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.subplots_adjust(bottom=0.15, left=0.13, right=0.94, top=0.95, wspace=0.2, hspace=0.2)
        ax.set_ylabel('cumulative propability')
        plt.savefig('%s/gradident_distrition_cdf_%s_%s.pdf' % (OUTPUTPATH, dnn, compressor))
    else:
        ax.set_xlim(left=-xlimit, right=xlimit)
        ax.set_ylabel('frequency')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
        if dnn== 'lenet':
            ax.set_ylim(top=1500)
        plt.subplots_adjust(bottom=0.15, left=0.13, right=0.97, top=0.95, wspace=0.2, hspace=0.2)
        plt.savefig('%s/gradident_distrition_%s_%s.pdf' % (OUTPUTPATH, dnn, compressor))
    plt.show()


def layerwise_distribution():
    fig, ax= plt.subplots(1, 1,figsize=(5.5,4.0))
    PLOT_THRES=False

    compressor='gaussion'
    compressor='none'
    dnn='resnet20';bs=32;lr=0.1;xlimit=0.023 if compressor=='none' else 0.18
    #dnn='vgg16';bs=128;lr=0.1;xlimit=0.0025 if compressor=='none' else 0.015
    #dnn='fcn5net';bs=128;lr=0.01;xlimit=0.01 if compressor=='none' else 0.1
    #dnn='lenet';bs=128;lr=0.01;xlimit=0.04 if compressor=='none' else 0.2
    #dnn='lstm';bs=20;lr=22;xlimit=0.00022 if compressor=='none' else 0.002
    #dnn='lstman4';bs=4;lr=0.0002;xlimit=0.38 if compressor=='none' else 3.5
    #LOGHOME='/datasets/shshi/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-thres-512000kbytes/%s-n8-bs32-lr0.1000-ns1-ds0.001' % (compressor, dnn);
    if compressor == 'none':
        LOGHOME='./logs/iclr/gradients/allreduce-gwarmup-dc1-model-iclr-gradients-thres-0kbytes/%s-n8-bs%d-lr%.4f-ns1-ds1.0' % (dnn, bs, lr);
    else:
        LOGHOME='./logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-debug-thres-512000kbytes/%s-n8-bs%d-lr%.4f-ns1-ds0.001' % (compressor, dnn, bs, lr);
    allfiles = [[] for i in range(10)]
    layer_dict = {}
    iter_dict = {}
    for file in os.listdir(LOGHOME):
        if file.endswith(".npy"):
            allfiles.append(file)
            layername = file.split('::')[1]
            iteration = int(file.split('r0_gradients_iter_')[1].split('::')[0])
            if file.find('bias') >= 0 or file.find('bn') >= 0 :
                continue
            if layername not in layer_dict:
                layer_dict[layername] = []
            if iteration not in iter_dict:
                iter_dict[iteration] = []
            layer_dict[layername].append(file)
            iter_dict[iteration].append(file)
    iterations=range(100, 2800) #[1000, 1100, 1200, 1900]
    #iterations=[1000, 1100, 1200, 1900]
    idx=0
    for i in iterations:
        if i % 200 != 0:
            continue
        #fn = '%s/gradients/%s/%s/r0_gradients_iter_%d.npy' % (LOGHOME, type, dnn, i)
        #fn = '%s/r0_gradients_iter_%d.npy' % (LOGHOME, i)
        #files = iter_dict[i][0:10]
        files = iter_dict[i][-2:]
        print('files: ', files)
        for f in files:
            layer_idx = int(f.split('::')[2].split('.npy')[0])
            fn = os.path.join(LOGHOME, f)
            print('fn: ', fn)
            grad = np.load(fn).flatten()
            l2norm = np.linalg.norm(grad)
            grad /= l2norm
            d = len(grad)
            k = int(0.001*d)
            print('# of params: ', d)
            if d < 1000:
                continue
            #count, bins, _ = ax.hist(grad, 200, density=False, alpha=1.0, label='iter%d-n%d-layer%d'% (i, d,layer_idx), histtype='step', linewidth=2)
            count, bins, _ = ax.hist(grad, 100, density=False, alpha=1.0, label='iter%d-n%d-layer%d'% (i, d,layer_idx), histtype='step', linewidth=1)
            if PLOT_THRES:
                abs_grad = np.abs(grad)
                sorted_grad = np.sort(abs_grad)[::-1]
                thres = sorted_grad[k]
                y = range(0, int(np.mean(count)*2))
                x = len(y)*[thres]
                ax.plot(x,y, color=colors[idx])
            idx+=1
    ax.grid(linestyle=':')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax.set_xlabel('gradient value')
    ax.set_ylabel('frequency')
    FONTSIZE=14
    ax.legend(fontsize=FONTSIZE, loc=1)
    #ax.set_xlim(left=-xlimit, right=xlimit)
    #if dnn== 'lenet':
    #    ax.set_ylim(top=1500)
    u.update_fontsize(ax, FONTSIZE)
    plt.subplots_adjust(bottom=0.15, left=0.13, right=0.97, top=0.95, wspace=0.2, hspace=0.2)
    #plt.savefig('%s/gradident_distrition_%s.pdf' % (OUTPUTPATH, dnn), bbox_inches='tight')
    #plt.savefig('%s/layerwise_gradident_distrition_%s_%s.pdf' % (OUTPUTPATH, dnn, compressor))
    plt.show()



def plot_sorting():
    #compressor='topk'
    fig, ax= plt.subplots(1, 1,figsize=(5.5,4.0))

    #compressor='gaussion'
    compressor='topk'
    dnn='resnet20';bs=32;lr=0.1;xlimit=0.18
    dnn='vgg16';bs=128;lr=0.1;xlimit=0.015
    dnn='fcn5net';bs=128;lr=0.01;xlimit=0.1
    dnn='lenet';bs=128;lr=0.01;xlimit=0.2
    dnn='lstm';bs=20;lr=22;xlimit=0.002
    #dnn='lstman4';bs=4;lr=0.0002;xlimit=3.5
    #LOGHOME='/datasets/shshi/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-convergence-thres-512000kbytes/%s-n8-bs32-lr0.1000-ns1-ds0.001' % (compressor, dnn);
    LOGHOME='../p2p-iclr/logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-rebuttal-logging-thres-512000kbytes/%s-n8-bs%d-lr%.4f-ns1-ds0.001' % (compressor, dnn, bs, lr);
    #LOGHOME='./logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-debug-thres-512000kbytes/%s-n8-bs%d-lr%.4f-ns1-ds0.001' % (compressor, dnn, bs, lr);
    #LOGHOME='./logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-debug-thres-512000kbytes/%s-n8-bs%d-lr0.1000-ns1-ds0.001' % (compressor, dnn, bs);
    iterations=range(1400, 1600) #[1000, 1100, 1200, 1900]
    r = 0.001
    PLOT_THRES=False
    idx=0
    def _line(x, d):
        return -1.0/d * x + 1
    for i in iterations:
        if i % 200 != 0:
            continue
        #fn = '%s/gradients/%s/%s/r0_gradients_iter_%d.npy' % (LOGHOME, type, dnn, i)
        fn = '%s/r0_gradients_iter_%d.npy' % (LOGHOME, i)
        #grad = np.load(fn).flatten()
        #d = len(grad)
        #d = 40
        d = 100000
        grad = np.random.normal(0, 5.0, size=d)
        #grad = np.random.uniform(0, 1.0, size=d)
        abs_grad = np.abs(grad)
        #abs_grad = np.power(grad, 2)
        sorted_grad = np.sort(abs_grad)[::-1]
        #l2norm = np.linalg.norm(sorted_grad)
        normed_grad = sorted_grad/np.max(sorted_grad)
        normed_grad = np.power(normed_grad, 2)
        k = int(r * len(sorted_grad))
        #plt.plot(normed_grad, label='iter-%d' % i, color=colors[idx])
        ax.plot(normed_grad, label=r'$\pi^2$', color=colors[idx])
        x = np.arange(0, len(grad))
        y = _line(x, d)
        ax.plot(x, y, label='reference line', color=colors[idx], linestyle='dashed')


        if PLOT_THRES:
            for i in range(k):
                y = _line(i,d)
                box = plt.Rectangle([i, 0], 1, y, fill=False,edgecolor='b', linewidth=1)
                plt.gca().add_patch(box)
                #plt.plot(x,y, color=colors[idx])
        idx+=1

    FONTSIZE=14
    ax.set_ylabel('normalized value')
    ax.set_xlabel(r'$i$')
    ax.legend(fontsize=FONTSIZE, loc=1)
    ax.grid(False)
    u.update_fontsize(ax, FONTSIZE)
    #plt.title(dnn)
    #plt.savefig('%s/absnormed.pdf' % (OUTPUTPATH), bbox_inches='tight')
    #plt.savefig('%s/areaillustration.pdf' % (OUTPUTPATH), bbox_inches='tight')
    plt.show()

def plot_generated_norm():
    fig, ax= plt.subplots(1, 1,figsize=(5.5,4.0))
    d = 100000
    grad = np.random.normal(0, 5.0, size=d)
    grad /= np.max(np.abs(grad))
    count, bins, _ = ax.hist(grad, 200, normed=False, alpha=1.0, color=colors[0])
    ax.set_ylabel('frequency')
    ax.set_xlabel(r'$\pi_{(i)}$')
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    FONTSIZE=12
    u.update_fontsize(ax, FONTSIZE)
    plt.savefig('%s/gennorm.pdf' % (OUTPUTPATH), bbox_inches='tight')
    plt.show()

def compared_topk_bounds():
    fig, ax= plt.subplots(1, 1,figsize=(5.5,4.0))
    d = 100000
    grad = np.random.normal(0, 1.0, size=d)
    USE_DNN = True

    if USE_DNN:
        compressor='topk'
        dnn='resnet20';bs=32;lr=0.1;xlimit=0.18
        dnn='fcn5net';bs=128;lr=0.01;xlimit=0.1
        #dnn='lstm';bs=20;lr=22;xlimit=0.002
        LOGHOME='./logs/iclr/gradients/allreduce-comp-%s-gwarmup-dc1-model-iclr-debug-thres-512000kbytes/%s-n8-bs%d-lr%.4f-ns1-ds0.001' % (compressor, dnn, bs, lr);
        fn = '%s/r0_gradients_iter_1000.npy' % (LOGHOME)
        grad = np.load(fn).flatten()

    d = len(grad)
    print('d: ', d)
    abs_grad = np.abs(grad)
    sorted_grad = np.sort(abs_grad)[::-1]
    ks = np.arange(0.01, 0.5, 0.01)
    xnorm = np.linalg.norm(sorted_grad)
    reals = []
    ours = []
    previous = []
    for r in ks: 
        k = int(r*d)
        topk = sorted_grad[0:k]
        topknorm = np.linalg.norm(topk)
        realbound = (xnorm**2 - 2*topknorm*xnorm + topknorm**2)/xnorm**2
        reals.append(realbound)
        ourbound = (1-k*1.0/d)**2
        ours.append(ourbound)
        previousbound = (1-k*1.0/d)
        previous.append(previousbound)

    ax.plot(ks, reals, label='exact value', color=colors[-1])
    ax.plot(ks, previous, label='previous studies', color=colors[-2], linestyle='dashdot')
    ax.plot(ks, ours, label='ours', color=colors[-3], linestyle='dotted')
    FONTSIZE=17
    ax.set_xlabel('k/d')
    ax.set_ylabel(r'$||u-Top_k(u)||^2 / ||u||^2 bound$ ')
    #ax.legend(fontsize=FONTSIZE, loc=1)
    u.update_fontsize(ax, FONTSIZE)
    if USE_DNN:
        plt.savefig('%s/topkbounds_%s.pdf' % (OUTPUTPATH, dnn), bbox_inches='tight')
    else:
        plt.savefig('%s/topkbounds.pdf' % (OUTPUTPATH), bbox_inches='tight')
    plt.show()


#layerwise_distribution()
#fit_distribution()
plot_sorting()
#plot_generated_norm()
#compared_topk_bounds()

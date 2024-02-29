import os
import time
import matplotlib.pyplot as plt
from plot_loss import plot_loss, markers, ax, fig, OUTPUTPATH
import itertools


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
    plot_loss(logfile, l, isacc=True, title=dnn) 


def icdcs_camerarady_vgg16():
    network='vgg16'; bs=128
    density=0.1;gpu='gpu21';compressor='gtopk'
    plot_with_params(network, 32, bs, 0.1, gpu, r'%s 32 GPUs, c=%d'%(compressor, 1/density), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)
    compressor='topk'
    plot_with_params(network, 32, bs, 0.1, gpu, r'%s 32 GPUs, c=%d'%(compressor, 1/density), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)


def icdcs_camerarady():
    network='resnet20'; bs=32
    #network='vgg16'; bs=128
    #plot_with_params(network, 32, bs, 0.1, 'MGD', r'AllReduce 32 GPUs', prefix='allreduce-baseline-gwarmup-dc1-model-debug-awu7', nsupdate=1, force_legend=True)
    #plot_with_params(network, 32, bs, 0.1, 'MGD', r'Top-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-debug-awu7', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #ds = [0.002, 0.005, 0.01, 0.05, 0.1]
    #density=0.001
    #for density in ds:
    #    gpu='gpu13'
    #    if density >= 0.05:
    #        gpu='gpu21'
    #    compressor='topk'
    #    plot_with_params(network, 32, bs, 0.1, gpu, r'%s 32 GPUs, c=%d'%(compressor, 1/density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-cr-verify', nsupdate=1, sg=2.5, density=density, force_legend=True)
    #    compressor='gtopk'
    #    if density == 0.01:
    #        plot_with_params('resnet20', 32, bs, 0.1, 'gpu13', r'%s 32 GPUs, c=%d'%(compressor, 1/density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full5', nsupdate=1, sg=2.5, density=density, force_legend=True)
    #    else:
    #        plot_with_params(network, 32, bs, 0.1, gpu, r'%s 32 GPUs, c=%d'%(compressor, 1/density), prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-icdcs-cr-verify', nsupdate=1, sg=2.5, density=density, force_legend=True)

    network='resnet20'; bs=32
    compressor='topk';density=0.001;
    plot_with_params(network, 32, bs, 0.1, 'MGD', r'Top-$k$,$\rho$=%.3f,$B$=%d'%(density,bs*32), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-debug-awu7', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    compressor='gtopk';density=0.001
    plot_with_params(network, 32, bs, 0.1, 'gpu13', r'gTop-$k$,$\rho$=%.3f,$B$=%d'%(density,bs*32), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr3'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)

    #density=5e-05;compressor='topk'
    #plot_with_params(network, 32, bs, 0.1, 'gpu15', r'%s 32 GPUs, c=%d'%(compressor, 1/density), prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-cr-verify', nsupdate=1, sg=2.5, density=density, force_legend=True)
    #density=5e-05;compressor='gtopk'
    #plot_with_params(network, 32, bs, 0.1, 'gpu15', r'%s 32 GPUs, c=%d'%(compressor, 1/density), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)

    #density=0.001; bs=4; compressor='topk'
    #plot_with_params(network, 32, bs, 0.1, 'gpu15', r'Top-$k$,$\rho$=%.3f,$B$=%d'%(density,bs*32), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)
    #density=0.001; bs=4; compressor='gtopk'
    #plot_with_params(network, 32, bs, 0.1, 'gpu15', r'gTop-$k$,$\rho$=%.3f,$B$=%d'%(density,bs*32), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)

    #density=0.002; bs=4; compressor='gtopk'
    #plot_with_params(network, 32, bs, 0.1, 'gpu19', r'%s 32 GPUs, c=%d, bs=%d'%(compressor, 1/density, bs), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)
    #density=0.002; bs=4; compressor='topk'
    #plot_with_params(network, 32, bs, 0.1, 'gpu19', r'%s 32 GPUs, c=%d, bs=%d'%(compressor, 1/density, bs), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)
    #density=3.125e-05; bs=4; compressor='gtopk'
    #plot_with_params(network, 32, bs, 0.1, 'gpu19', r'%s 32 GPUs, c=%d, bs=%d'%(compressor, 1/density, bs), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)
    #density=5e-04; bs=4; compressor='gtopk'
    #plot_with_params(network, 32, bs, 0.1, 'gpu19', r'%s 32 GPUs, c=%d, bs=%d'%(compressor, 1/density, bs), prefix='allreduce-comp-%s-baseline-gwarmup-dc1-model-icdcs-cr-verify'%compressor, nsupdate=1, sg=2.5, density=density, force_legend=True)

def vgg16_icdcs():
    pass

def resnet20():
    plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'AllReduce 32 GPUs', prefix='allreduce-baseline-gwarmup-dc1-model-debug-awu7', nsupdate=1, force_legend=True)
<<<<<<< HEAD
    plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'Top-$k$ 32 GPUs', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-debug-awu7', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'gTop-$k$ 32 GPUs', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-awu7', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'gTop-$k$ 32 GPUs', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-awu8', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'gTop-$k$ 32 GPUs', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-awu9', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'gTop-$k$ 32 GPUs', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-b1', nsupdate=1, sg=2.5, density=0.001, force_legend=True)

=======
    plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'Top-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-debug-awu7', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'Top-$k$ 32 GPUs, c=100', prefix='allreduce-comp-topk-baseline-gwarmup-dc1-model-icdcs-cr3', nsupdate=1, sg=2.5, density=0.01, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu16', r'Top-$k$ and localTop-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-topk2-baseline-gwarmup-dc1-model-debug-full', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu16', r'Top-$k$ + localTop-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-topk2-baseline-gwarmup-dc1-model-debug-full2', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'gTop-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-awu8', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu16', r'gTop-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full2', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu16', r'gTop-$k$2 32 GPUs, c=1000', prefix='allreduce-comp-gtopk2-baseline-gwarmup-dc1-model-debug-full2', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu17', r'gTop-$k$2 32 GPUs, c=1000', prefix='allreduce-comp-gtopk2-baseline-gwarmup-dc1-model-debug-full3', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu17', r'gTop-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full3', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu16', r'gTop-$k$(4) 32 GPUs, c=1000', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full4', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    plot_with_params('resnet20', 32, 32, 0.1, 'gpu16', r'gTop-$k$(5) 32 GPUs, c=1000', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full5', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu19', r'MeanGTop-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-meangtopk-baseline-gwarmup-dc1-model-icdcs-cr', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'MeanGTop-$k$ (Residual Mean) 32 GPUs, c=1000', prefix='allreduce-comp-meangtopk-baseline-gwarmup-dc1-model-icdcs-cr2', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'MeanGTop-$k$ (Total Mean) 32 GPUs, c=1000', prefix='allreduce-comp-meangtopk-baseline-gwarmup-dc1-model-icdcs-cr3', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu16', r'gTop-$k$(6) 32 GPUs, c=200', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full5', nsupdate=1, sg=2.5, density=0.005, force_legend=True)
    plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'gTop-$k$(7) 32 GPUs, c=100', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full5', nsupdate=1, sg=2.5, density=0.01, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'gTop-$k$(8) 32 GPUs, c=33', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full5', nsupdate=1, sg=2.5, density=0.03, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'Layerwise gTop-$k$ 32 GPUs, c=33', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-layerwise', nsupdate=1, sg=2.5, density=0.03, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu15', r'Layerwise gTop-$k$ 32 GPUs, c=1000', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-layerwise', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'Layerwise gTop-$k$+MC 32 GPUs, c=33', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-layerwise-mc', nsupdate=1, sg=2.5, density=0.03, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'gTop-$k$+MC 32 GPUs, c=33', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full5-mc', nsupdate=1, sg=2.5, density=0.03, force_legend=True)
>>>>>>> 3a7804b462e4a4dd427865af26f16f8d03653255
    #plot_with_params('resnet20', 32, 32, 0.1, 'MGD', r'gTop-$k$ 32 GPUs', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-awu9', nsupdate=1, sg=2.5, density=0.001, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'gTop-$k$ 32 GPUs, c=200', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full', nsupdate=1, sg=2.5, density=0.005, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'gTop-$k$ 32 GPUs, c=100', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full', nsupdate=1, sg=2.5, density=0.01, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'gTop-$k$ 32 GPUs, c=33', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full', nsupdate=1, sg=2.5, density=0.03, force_legend=True)
    #plot_with_params('resnet20', 32, 32, 0.1, 'gpu13', r'gTop-$k$ 32 GPUs, c=25', prefix='allreduce-comp-gtopk-baseline-gwarmup-dc1-model-debug-full', nsupdate=1, sg=2.5, density=0.04, force_legend=True)

def vgg16():
    pass

def resnet50():
    pass

if __name__ == '__main__':
    #resnet20()
    #icdcs_camerarady_vgg16()
    icdcs_camerarady()
    ax.legend(loc=4)
    plt.subplots_adjust(bottom=0.18, left=0.2, right=0.96, top=0.9)
    plt.savefig('%s/%s_acc.pdf' % (OUTPUTPATH, 'resnet20'))
    plt.show()

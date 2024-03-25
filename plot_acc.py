import os
import time
import matplotlib.pyplot as plt
from plot_loss import plot_loss, markers, ax, fig
import itertools
'''
def plot_with_params(dnn, nworkers, bs, lr, prefix='', title='ResNet-20'):
    dir = '%s-n%d-bs%d-lr%.4f' % (dnn, nworkers, bs, lr)
    if prefix != '':
        logfile = os.path.join('weights', prefix, dir, 'evaluate.log')
    else:
        logfile = os.path.join('weights', dir, 'evaluate.log')
    print logfile
    legend = dir + ' ' + prefix
    if prefix == 'baseline':
        legend = dir + ' Lian et al. 2018'
    if nworkers == 1:
        legend = 'bs=32 SGD 1 GPU' 
    plot_loss(logfile, legend, isacc=True, title=dnn)
'''
def plot_with_params(dnn, nworkers, overlap, lr, prefix='', title='ResNet-20',logdirlist=[]):
    overlapping = 'overlap' if overlap == True else 'nonoverlap'
    dir = '%s-n%d-%s.4f' % (dnn, nworkers, overlapping)
    plot_loss(logfile, legend, isacc=True, title=dnn)

def resnet20():
    #plot_with_params('resnet20', nworkers=4, bs=32, lr=0.1, prefix='allreduce')
    plot_with_params('resnet20', nworkers=4, bs=32, lr=0.1, prefix='baseline-modelhpcl')
    plot_with_params('resnet20', nworkers=8, bs=32, lr=0.1, prefix='baseline-modelhpcl')
    plot_with_params('resnet20', nworkers=16, bs=32, lr=0.1, prefix='baseline-modelhpcl')

def vgg16():
    plot_with_params('vgg16', nworkers=4, bs=32, lr=0.1, prefix='baseline-modelhpcl')
    plot_with_params('vgg16', nworkers=8, bs=32, lr=0.1, prefix='baseline-modelhpcl')
    plot_with_params('vgg16', nworkers=16, bs=32, lr=0.1, prefix='baseline-modelhpcl')
    plot_with_params('vgg16', nworkers=16, bs=32, lr=0.0005, prefix='baseline-modelhpcl')


def resnet50():
    pass

if __name__ == '__main__':
    interval = 2
    while True:
        ax.clear()
        vgg16()
        #resnet20()
        ax.legend(loc=4)
        plt.pause(interval)
        plt.draw()
        time.sleep(interval)
        break
    plt.show()

# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import time
import psutil

import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.utils.data.distributed
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as ct
import settings
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True
from settings import logger, formatter
import struct
import models
import logging
import utils
import math
import json
from LR import LRSchedule
from encoding import huffman
#from tensorboardX import SummaryWriter
#from datasets import DatasetHDF5
from profiling import benchmark
import transformer.Constants as Constants
#writer = SummaryWriter()

import ptb_reader
import models.lstm as lstmpy
from torch.autograd import Variable

#from data_sampler import CachedIndexImages, CachedSampler, CachedImageFolder

if settings.FP16:
    import apex
else:
    apex = None

#torch.manual_seed(0)
torch.set_num_threads(1)

if settings.EFFICIENT_IO:
    NUM_CPU_THREADS=1
else:
    NUM_CPU_THREADS=8

_support_datasets = ['imagenet', 'cifar10', 'an4', 'ptb', 'wt2', 'mnist', 'wmt2016']
_support_dnns = ['alexnet', 'alexnetbn',
        'resnet18', 'resnet50', 'resnet101', 'resnet152', 
        'densenet121', 'densenet161', 'densenet201', 
        'googlenet', 'inceptionv4', 'inceptionv3', 
        'vgg16i', 
        'resnet20', 'resnet56', 'resnet110', 
        'vgg19', 'vgg16',  
        'lstman4', 
        'lstm', 'lstmwt2',
        'mnistnet', 'fcn5net', 'lenet', 
        'lr',
        'transformer']


def init_processes(rank, size, backend='tcp', master='gpu10'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master 
    os.environ['MASTER_PORT'] = '5935'

    #master_ip = "gpu20"
    #master_mt = '%s://%s:%s' % (backend, master_ip, '5955')
    logger.info("initialized trainer rank: %d of %d......" % (rank, size))
    #dist.init_process_group(backend=backend, init_method=master_mt, rank=rank, world_size=size)
    dist.init_process_group(backend=backend, rank=rank, world_size=size)
    logger.info("finished trainer rank: %d......" % rank)

class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.name = 'mnistnet'

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def get_available_gpu_device_ids(ngpus):
    return range(0, ngpus)

def create_net(num_classes, dnn='resnet20', dataset='cifar10', **kwargs):
    ext = None
    if dnn in ['resnet20', 'resnet56', 'resnet110']:
        net = models.__dict__[dnn](num_classes=num_classes)
    elif dnn == 'vgg16':
        net = models.VGG(dnn.upper())
    elif dnn == 'alexnet':
        net = torchvision.models.alexnet()
    elif dnn == 'alexnetbn':
        net = models.AlexNetBN()
    # elif dnn == 'resnet18':
    #     net = torchvision.models.resnet18(num_classes=num_classes)
    # elif dnn == 'resnet50':
    #     net = torchvision.models.resnet50(num_classes=num_classes)
    # elif dnn == 'resnet101':
    #     net = torchvision.models.resnet101(num_classes=num_classes)
    # elif dnn == 'resnet152':
    #     net = torchvision.models.resnet152(num_classes=num_classes)
    # """
    #     For cifar10
    # """
    elif dnn == 'resnet18' and dataset == "cifar10":
        net = models.cifar_resnet18(num_classes=num_classes)
    elif dnn == 'resnet50' and dataset == "cifar10":
        net = models.cifar_resnet50(num_classes=num_classes)
    elif dnn == 'resnet101' and dataset == "cifar10":
        net = models.cifar_resnet101(num_classes=num_classes)
    elif dnn == 'resnet152' and dataset == "cifar10":
        net = models.cifar_resnet152(num_classes=num_classes)
    elif dnn == 'resnet18' and dataset == "imagenet":
        net = torchvision.models.resnet18(num_classes=num_classes)
    elif dnn == 'resnet50' and dataset == "imagenet":
        net = torchvision.models.resnet50(num_classes=num_classes)
    elif dnn == 'resnet101' and dataset == "imagenet":
        net = torchvision.models.resnet101(num_classes=num_classes)
    elif dnn == 'resnet152' and dataset == "imagenet":
        net = torchvision.models.resnet152(num_classes=num_classes)

    elif dnn == 'densenet121':
        net = torchvision.models.densenet121(num_classes=num_classes)
    elif dnn == 'densenet161':
        net = torchvision.models.densenet161(num_classes=num_classes)
    elif dnn == 'densenet201':
        net = torchvision.models.densenet201(num_classes=num_classes)
    elif dnn == 'inceptionv4':
        net = models.inceptionv4(num_classes=num_classes)
    elif dnn == 'inceptionv3':
        net = torchvision.models.inception_v3(num_classes=num_classes)
    elif dnn == 'vgg16i': # vgg16 for imagenet
        net = torchvision.models.vgg16(num_classes=num_classes)
    elif dnn == 'googlenet':
        net = models.googlenet()
    elif dnn == 'mnistnet':
        net = MnistNet()
    elif dnn == 'fcn5net':
        net = models.FCN5Net()
    elif dnn == 'lenet':
        net = models.LeNet()
    elif dnn == 'lr':
        net = models.LinearRegression()
    elif dnn == 'lstman4':
        net, ext = models.LSTMAN4(datapath=kwargs['datapath'])
    elif dnn == 'lstm':
        net = lstmpy.lstm(vocab_size=kwargs['vocab_size'], batch_size=kwargs['batch_size'], dp_keep_prob=0.3)
    elif dnn == 'lstmwt2':
        net = lstmpy.lstmwt2(vocab_size=kwargs['vocab_size'], batch_size=kwargs['batch_size'], dp_keep_prob=0.5)
    elif dnn == 'transformer':
        from transformer.Models import Transformer
        net = Transformer(
            kwargs['src_vocab_size'],
            kwargs['tgt_vocab_size'],
            kwargs['max_token_seq_len'],
            tgt_emb_prj_weight_sharing=False,
            emb_src_tgt_weight_sharing=False)
    else:
        errstr = 'Unsupport neural network %s' % dnn
        logger.error(errstr)
        raise errstr 
    return net, ext


class DLTrainer:

    def __init__(self, rank, size, master='gpu10', localsgd=False, dist=True, ngpus=1, batch_size=32, 
        is_weak_scaling=True, data_dir='./data', dataset='cifar10', dnn='resnet20', 
        lr=0.04, nworkers=1, prefix=None, sparsity=0.95, pretrain=None, num_steps=35, tb_writer=None, amp_handle=None,optimizer_name='SGD'):

        self.size = size
        self.rank = rank
        self.pretrain = pretrain
        self.dataset = dataset
        self.prefix=prefix
        self.num_steps = num_steps
        self.ngpus = ngpus
        self.writer = tb_writer
        self.amp_handle = amp_handle
        self.optimizer_name = optimizer_name
        self.localsgd = localsgd
        if settings.EFFICIENT_IO:
            self.cached_index_images = CachedIndexImages()
        else:
            self.cached_index_images = None

        if self.ngpus > 0:
            self.batch_size = batch_size * self.ngpus if is_weak_scaling else batch_size
        else:
            self.batch_size = batch_size
        self.num_batches_per_epoch = -1
        if self.dataset == 'cifar10' or self.dataset == 'mnist':
            self.num_classes = 10
        elif self.dataset == 'imagenet':
            self.num_classes = 1000
        elif self.dataset == 'an4':
            self.num_classes = 29 
        elif self.dataset in ['ptb', 'wt2']:
            self.num_classes = 10
        elif self.dataset == 'wmt2016':
            self.num_classes = 10
        self.nworkers = nworkers # just for easy comparison
        self.data_dir = data_dir
        if type(dnn) != str:
            self.net = dnn
            self.dnn = dnn.name
            self.ext = None # leave for further parameters
        else:
            self.dnn = dnn
            # TODO: Refact these codes!
            if self.dnn in ['lstm', 'lstmwt2']:
                if data_dir is not None:
                    self.data_prepare()
                self.net, self.ext = create_net(self.num_classes, self.dnn, self.dataset, vocab_size=self.vocab_size, batch_size=self.batch_size)
            elif self.dnn == 'lstman4':
                self.net, self.ext = create_net(self.num_classes, self.dnn, self.dataset, datapath=self.data_dir)
                if data_dir is not None:
                    self.data_prepare()
            elif self.dnn == 'transformer':
                if data_dir is not None:
                    self.data_prepare()
                self.net, self.ext = create_net(self.num_classes, self.dnn, 
                        self.dataset, 
                        datapath=self.data_dir,
                        src_vocab_size=self.src_vocab_size,
                        tgt_vocab_size=self.tgt_vocab_size,
                        max_token_seq_len=self.max_token_seq_len)
            else:
                if data_dir is not None:
                    self.data_prepare()
                self.net, self.ext = create_net(self.num_classes, self.dnn, self.dataset)
        self.lr = lr
        self.base_lr = self.lr
        self.is_cuda = self.ngpus > 0
        #if self.is_cuda:
        #    torch.cuda.manual_seed_all(3000)

        if self.is_cuda:
            if self.ngpus > 1:
                devices = get_available_gpu_device_ids(ngpus)
                self.net = torch.nn.DataParallel(self.net, device_ids=devices).cuda()
            else:
                self.net.cuda()
        self.net.share_memory()
        self.accuracy = 0
        self.loss = 0.0
        self.train_iter = 0
        self.recved_counter = 0
        self.master = master
        self.average_iter = 0
        if dist:
            init_processes(rank, size, master=master)
        if self.dataset != 'an4':
            if self.is_cuda:
                self.criterion = nn.CrossEntropyLoss().cuda()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            from warpctc_pytorch import CTCLoss
            self.criterion = CTCLoss()
        self.lr_scheduler = getattr(LRSchedule, 'linear')(lr_init=self.lr, epochs=settings.MAX_EPOCHS, extra=0)
        weight_decay = 1e-4
        self.m = 0.9 # momentum
        nesterov = False
        if self.dataset == 'an4':
            #nesterov = True
            self.lstman4_lr_epoch_tag = 0
            #weight_decay = 0.
        elif self.dataset in ['ptb', 'wt2']:
            self.m = 0
            weight_decay = 0
        elif self.dataset == 'imagenet':
            pass
            #if self.dnn == 'alexnetbn':
            #    weight_decay = 0.0005 # default for alexnetbn
            #else:
            #    weight_decay = 1e-4 # default for resnet50
            #self.m = 0.875
            #weight_decay = 2*3.0517578125e-05

        #decay = []
        #no_decay = []
        #for name, param in self.net.named_parameters():
        #    if not param.requires_grad:
        #        continue
        #    if len(param.shape) == 1 or 'bn' in name or 'bias' in name:
        #        no_decay.append(param)
        #    else:
        #        decay.append(param)
        #parameters = [{'params': no_decay, 'weight_decay': 0.},
        #            {'params': decay, 'weight_decay': weight_decay}]
        if (self.optimizer_name == 'Adam'):
            self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        elif(self.optimizer_name == 'AdamW'):
            self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        elif(self.optimizer_name == 'SGD'):
            self.optimizer = optim.SGD(self.net.parameters(), 
                lr=self.lr,
                momentum=self.m, 
                weight_decay=weight_decay,
                nesterov=nesterov)

        self.train_epoch = 0

        if self.pretrain is not None and os.path.isfile(self.pretrain):
            self.load_model_from_file(self.pretrain)

        self.sparsities = []
        self.compression_ratios = []
        self.communication_sizes = []
        self.remainer = {}
        self.v = {} # 
        self.target_sparsities = [1.]
        self.sparsity = sparsity
        logger.info('target_sparsities: %s', self.target_sparsities)
        self.avg_loss_per_epoch = 0.0
        self.timer = 0.0
        self.forwardtime = 0.0
        self.backwardtime = 0.0
        self.backwardtime_tmp = 0.0
        self.iotime = 0.0
        self.epochs_info = []
        self.distributions = {}
        self.gpu_caches = {}
        self.delays = []
        self.num_of_updates_during_comm = 0 
        self.train_acc_top1 = []
        if apex is not None:
            self.init_fp16()
        logger.info('num_batches_per_epoch: %d'% self.num_batches_per_epoch)

    def init_fp16(self):
        model, optim = apex.amp.initialize(self.net, self.optimizer, opt_level='O2', loss_scale=128.0)
        self.net = model
        self.optimizer = optim

    def get_acc(self):
        return self.accuracy

    def get_loss(self):
        return self.loss

    def get_model_state(self):
        return self.net.state_dict()

    def get_data_shape(self):
        return self._input_shape, self._output_shape

    def get_train_epoch(self):
        return self.train_epoch

    def get_train_iter(self):
        return self.train_iter

    def set_train_epoch(self, epoch):
        self.train_epoch = epoch

    def set_train_iter(self, iteration):
        self.train_iter = iteration

    def load_model_from_file(self, filename):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['state'])
        self.train_epoch = checkpoint['epoch']
        self.train_iter = checkpoint['iter']
        lr = checkpoint.get('lr', None)
        if lr:
            self.lr = lr
        logger.info('Load pretrain model: %s, start from epoch %d and iter: %d', filename, self.train_epoch, self.train_iter)

    def get_num_of_training_samples(self):
        return len(self.trainset)

    def imagenet_prepare(self):
        # Data loading code
        traindir = os.path.join(self.data_dir, 'train')
        testdir = os.path.join(self.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if self.dnn == 'alexnetbn':
            image_size = 227
        else:
            image_size = 224
        #image_size = 128
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 1000)
        if settings.EFFICIENT_IO:
            trainset = CachedImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ]), cached_index_images=self.cached_index_images)
        else:
            #hdf5fn = os.path.join(self.data_dir, 'imagenet-shuffled.hdf5')
            #hdf5fn = os.path.join(self.data_dir, 'imagenet-2012.hdf5')
            trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
            #trainset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([
            #    transforms.ToPILImage(),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ]))#, fake=settings.FAKE_DATA)
        self.trainset = trainset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            if settings.EFFICIENT_IO:
                train_sampler = CachedSampler(self.trainset, num_replicas=self.nworkers, 
                        rank=self.rank, cached_index_images=self.cached_index_images)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=True, sampler=train_sampler)
        testset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([
        #testset = DatasetHDF5(hdf5fn, 'val', transforms.Compose([
        #        transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]))#, fake=settings.FAKE_DATA)

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=8, pin_memory=True)

    def cifar10_prepare(self):
        #transform = transforms.Compose(
        #    [transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #train_transform = transform
        #test_transform = transform
        image_size = 32
        self._input_shape = (self.batch_size, 3, image_size, image_size)
        self._output_shape = (self.batch_size, 10)
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                               download=True, transform=test_transform)
        self.trainset = trainset
        self.testset = testset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        print("Downlaod the CIFAR10 dataset.")
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                                 shuffle=False, num_workers=8)
        self.classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def mnist_prepare(self):
        trans = []
        if self.dnn == 'lenet':
            image_size = 32
            trans.append(transforms.Resize(32))
        else:
            image_size = 28
        trans.extend([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])
        self._input_shape = (self.batch_size, 1, image_size, image_size)
        self._output_shape = (self.batch_size, 10)

        trainset = torchvision.datasets.MNIST(self.data_dir, train=True, download=True,
                    transform=transforms.Compose(trans))
        self.trainset = trainset
        testset = torchvision.datasets.MNIST(self.data_dir, train=False, transform=transforms.Compose(trans))
        self.testset = testset
        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(trainset,
                batch_size=self.batch_size, shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
        self.testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=self.batch_size, shuffle=False, num_workers=1)

    def ptb_prepare(self):
        # Data loading code

        # =====================================
        # num_workers=NUM_CPU_THREADS num_workers=1
        # batch_size=self.batch_size
        # num_steps = 35
        # hidden_size = 1500

        # =================================
        prefix = self.dataset
        raw_data = ptb_reader.ptb_raw_data(data_path=self.data_dir, prefix=prefix)
        train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
        self.vocab_size = len(word_to_id)


        self._input_shape = (self.batch_size, self.num_steps)
        self._output_shape = (self.batch_size, self.num_steps)

        print('Vocabluary size: {}'.format(self.vocab_size))

        print('load data')

        epoch_size = ((len(train_data) // self.batch_size) - 1) // self.num_steps

        train_set = ptb_reader.TrainDataset(train_data, self.batch_size, self.num_steps)
        self.trainset = train_set
        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler
        self.trainloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=True, sampler=train_sampler, drop_last=False)


        test_set = ptb_reader.TestDataset(valid_data, self.batch_size, self.num_steps)
        #test_set = ptb_reader.TestDataset(test_data, self.batch_size, self.num_steps)
        self.testset = test_set
        self.testloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size, shuffle=False,
            num_workers=1, pin_memory=True)
        print('=========****** finish getting ptb data, num_of_training samples: %d ===========' % self.get_num_of_training_samples())

    def an4_prepare(self):
        from audio_data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
        from decoder import GreedyDecoder
        audio_conf = self.ext['audio_conf']
        labels = self.ext['labels']
        train_manifest = os.path.join(self.data_dir, 'an4_train_manifest.csv')
        val_manifest = os.path.join(self.data_dir, 'an4_val_manifest.csv')


        with open('labels.json') as label_file:
            labels = str(''.join(json.load(label_file)))
        trainset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=train_manifest, labels=labels, normalize=True, augment=True)
        self.trainset = trainset
        testset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=val_manifest, labels=labels, normalize=True, augment=False)
        self.testset = testset

        if self.nworkers > 1:
            train_sampler = DistributedBucketingSampler(self.trainset, batch_size=self.batch_size, num_replicas=self.nworkers, rank=self.rank)
        else:
            train_sampler = BucketingSampler(self.trainset, batch_size=self.batch_size)

        self.train_sampler = train_sampler
        trainloader = AudioDataLoader(self.trainset, num_workers=4, batch_sampler=self.train_sampler)
        testloader = AudioDataLoader(self.testset, batch_size=self.batch_size,
                                  num_workers=4)
        self.trainloader = trainloader
        self.testloader = testloader
        decoder = GreedyDecoder(labels)
        self.decoder = decoder

    def wmt2016_prepare(self):
        from transformer_dataset import TranslationDataset, paired_collate_fn
        data_file = os.path.join(self.data_dir, 'multi30k.atok.low.pt')
        data = torch.load(data_file)
        self.max_token_seq_len = data['settings'].max_token_seq_len

        trainset = TranslationDataset(
                    src_word2idx=data['dict']['src'],
                    tgt_word2idx=data['dict']['tgt'],
                    src_insts=data['train']['src'],
                    tgt_insts=data['train']['tgt'])
        self.trainset = trainset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size, collate_fn=paired_collate_fn, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=True, sampler=train_sampler)

        testset = TranslationDataset(
                    src_word2idx=data['dict']['src'],
                    tgt_word2idx=data['dict']['tgt'],
                    src_insts=data['valid']['src'],
                    tgt_insts=data['valid']['tgt'])
        self.testset = testset

        self.testloader = torch.utils.data.DataLoader(
                testset,
                num_workers=2,
                batch_size=self.batch_size,
                collate_fn=paired_collate_fn)

        self.src_vocab_size = self.trainloader.dataset.src_vocab_size
        self.tgt_vocab_size = self.trainloader.dataset.tgt_vocab_size

        self._input_shape = (self.batch_size, None)
        self._output_shape = (self.batch_size, None)

    def data_prepare(self):
        if self.dataset == 'imagenet':
            self.imagenet_prepare()
        elif self.dataset == 'cifar10':
            self.cifar10_prepare()
        elif self.dataset == 'mnist':
            self.mnist_prepare()
        elif self.dataset == 'an4':
            self.an4_prepare()
        elif self.dataset in ['ptb', 'wt2']:
            self.ptb_prepare()
        elif self.dataset == 'wmt2016':
            self.wmt2016_prepare()
        else:
            errstr = 'Unsupport dataset: %s' % self.dataset
            logger.error(errstr)
            raise errstr
        self.data_iterator = iter(self.trainloader)
        self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)//(self.batch_size*self.nworkers)
        #self.num_batches_per_epoch = self.get_num_of_training_samples()/(self.batch_size*self.nworkers)

    def update_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update_nworker(self, nworkers, new_rank=-1):
        if new_rank >= 0:
            rank = new_rank
            self.nworkers = nworkers
        else:
            reduced_worker = self.nworkers - nworkers
            rank = self.rank
            if reduced_worker > 0 and self.rank >= reduced_worker:
                rank = self.rank - reduced_worker
        self.rank = rank
        if self.dnn != 'lstman4':
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.trainset, num_replicas=nworkers, rank=rank)
            train_sampler.set_epoch(self.train_epoch)
            shuffle = False
            self.train_sampler = train_sampler
            self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                      shuffle=shuffle, num_workers=NUM_CPU_THREADS, sampler=train_sampler)
            self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                     shuffle=False, num_workers=1)
        self.nworkers = nworkers
        self.num_batches_per_epoch = (self.get_num_of_training_samples()+self.batch_size*self.nworkers-1)//(self.batch_size*self.nworkers)

    def data_iter(self):
        try:
            #d = self.data_iterator.next()
            d = next(self.data_iterator)
        except:
            self.data_iterator = iter(self.trainloader)
            d = next(self.data_iterator)
        if self.dnn in ['lstm', 'lstmwt2'] and d[0].size()[0] != self.batch_size:
            return self.data_iter()
        return d

    def _adjust_learning_rate_lstman4(self, progress, optimizer):
        #if settings.WARMUP and progress< 5:
        #    warmup_total_iters = self.num_batches_per_epoch * 5 
        #    min_lr = self.base_lr / self.nworkers
        #    lr_interval = (self.base_lr - min_lr) / warmup_total_iters
        #    self.lr = min_lr + lr_interval * self.train_iter
        #    #warmuplr = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
        #    #self.lr = warmuplr[progress]
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = self.lr
        #    return 
        if self.lstman4_lr_epoch_tag != progress:
            self.lstman4_lr_epoch_tag = progress 
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 1.01 
            self.lr = self.lr / 1.01

    def _adjust_learning_rate_lstmptb(self, progress, optimizer):
        #warmup = 2
        #if settings.WARMUP and progress < warmup:
        #    warmup_total_iters = self.num_batches_per_epoch * warmup
        #    min_lr = self.base_lr / warmup_total_iters 
        #    lr_interval = (self.base_lr - min_lr) / warmup_total_iters
        #    self.lr = min_lr + lr_interval * self.train_iter
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = self.lr
        #    return self.lr
        #first = 23
        #second = 60
        #third = 80
        #if progress < first: 
        #    lr = self.base_lr
        #elif progress < second: 
        #    lr = self.base_lr *0.1
        #elif progress < third:
        #    lr = self.base_lr *0.01
        #else:
        #    lr = self.base_lr *0.001
        #self.lr = lr
        self.lr = self.base_lr * (0.97**progress)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr 

    def _adjust_learning_rate_general(self, progress, optimizer):
        warmup = 5
        if settings.WARMUP and progress < warmup:
            warmup_total_iters = self.num_batches_per_epoch * warmup
            min_lr = self.base_lr / warmup_total_iters 
            lr_interval = (self.base_lr - min_lr) / warmup_total_iters
            self.lr = min_lr + lr_interval * self.train_iter
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
            return self.lr
        first = 81
        second = first + 41
        third = second+33
        if self.dataset == 'imagenet':
            first = 30
            second = 60
            third = 80
        elif self.dataset == 'ptb':
            first = 24
            second = 60
            third = 80
        if progress < first: #40:  30 for ResNet-50, 40 for ResNet-20
            #interval_iters = first * self.num_batches_per_epoch+2
            #lr_interval = (self.base_lr-self.base_lr*0.1)/interval_iters
            #lr = self.base_lr - (self.train_iter % interval_iters) * lr_interval
            lr = self.base_lr
        elif progress < second: #80: 70 for ResNet-50, 80 for ResNet-20
            #interval_iters = (second-first) * self.num_batches_per_epoch+2
            #lr_interval = (self.base_lr*0.1-self.base_lr*0.01)/interval_iters
            #lr = self.base_lr *0.1 - (self.train_iter % interval_iters) * lr_interval
            lr = self.base_lr * 0.1
        elif progress < third:
            #interval_iters = (third-second) * self.num_batches_per_epoch+2
            #lr_interval = (self.base_lr*0.01-self.base_lr*0.001)/interval_iters
            #lr = self.base_lr *0.01 - (self.train_iter % interval_iters) * lr_interval
            lr = self.base_lr * 0.01
        else:
            lr = self.base_lr *0.001
        #if self.train_iter % self.num_batches_per_epoch != 0:
        #    lr = lr - lr/(self.train_iter % self.num_batches_per_epoch+1)
        self.lr = lr
        if settings.ZHU:
            k = (self.train_iter+1)#*self.nworkers
            lr = 1.0/(np.sqrt(k) * np.log(k))
            max_lr = self.base_lr
            if lr > max_lr:
                lr = max_lr
            self.lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr 

    def _adjust_learning_rate_vgg16(self, progress, optimizer):
        if progress > 0 and progress % 25 == 0:
            self.lr = self.base_lr / (2**(progress/25))
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def _adjust_learning_rate_customized(self, progress, optimizer):
        def _get_increased_lrs(base_lr, min_epoch, max_epoch):
            npe = self.num_batches_per_epoch
            total_iters = (max_epoch-min_epoch)*npe
            min_lr = base_lr/total_iters
            lr_interval = (base_lr - min_lr) /total_iters 
            lr = min_lr + lr_interval * (self.train_iter-min_epoch*npe)
            return lr
        def _get_decreased_lrs(base_lr, target_lr, min_epoch, max_epoch):
            npe = self.num_batches_per_epoch
            total_iters = (max_epoch-min_epoch)*npe
            lr_interval = (base_lr-target_lr)/total_iters
            lr = base_lr - lr_interval * (self.train_iter-min_epoch*npe)
            return lr

        warmup = 10
        if settings.WARMUP and progress < warmup:
            self.lr = _get_increased_lrs(self.base_lr, 0, warmup)
            #self.lr = self.base_lr
        elif progress < 15:
            self.lr = self.base_lr
        #elif progress < 40:
        #    self.lr = _get_decreased_lrs(self.base_lr*0.1, self.base_lr*0.01, 20, 40)
        #elif progress < 27: 
        #    self.lr = self.base_lr*0.05#_get_increased_lrs(self.base_lr, 19, 30) 
        elif progress < 25:
            self.lr = self.base_lr*0.1 
        elif progress < 35:
            self.lr = self.base_lr*0.01
        else:
            self.lr = self.base_lr*0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def _adjust_learning_rate_cosine(self, progress, optimizer):
        def _get_increased_lrs(base_lr, min_epoch, max_epoch):
            npe = self.num_batches_per_epoch
            total_iters = (max_epoch-min_epoch)*npe
            min_lr = base_lr/total_iters
            lr_interval = (base_lr - min_lr) /total_iters 
            lr = min_lr + lr_interval * (self.train_iter-min_epoch*npe)
            return lr
        warmup = 14
        max_epochs = 40
        if settings.WARMUP and progress < warmup:
            self.lr = _get_increased_lrs(self.base_lr, 0, warmup)
        elif progress < max_epochs:
            e = progress - warmup 
            es = max_epochs - warmup 
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * self.base_lr
            self.lr = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def adjust_learning_rate(self, progress, optimizer):
        #if self.dnn == 'vgg16':
        #    return self._adjust_learning_rate_vgg16(progress, optimizer)
        if self.dnn == 'lstman4':
           return self._adjust_learning_rate_lstman4(self.train_iter//self.num_batches_per_epoch, optimizer)
        elif self.dnn in ['lstm', 'lstmwt2']:
            return self._adjust_learning_rate_lstmptb(progress, optimizer)
        return self._adjust_learning_rate_general(progress, optimizer)
        #return self._adjust_learning_rate_customized(progress, optimizer)
        #return self._adjust_learning_rate_cosine(progress, optimizer)

    def print_weight_gradient_ratio(self):
        # Tensorboard
        if self.rank == 0 and self.writer is not None:
            for name, param in self.net.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.train_epoch)
        return

    def finish(self):
        if self.writer is not None:
            self.writer.close()

    def cal_accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                #res.append(correct_k)
                res.append(correct_k.mul_(1.0 / batch_size))
            return res

    def cal_performance_transformer(self, pred, gold, smoothing=False):
        def _cal_loss(pred, gold, smoothing):
            ''' Calculate cross entropy loss, apply label smoothing if needed. '''
            gold = gold.contiguous().view(-1)
            if smoothing:
                eps = 0.1
                n_class = pred.size(1)
                one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
                log_prb = F.log_softmax(pred, dim=1)
                non_pad_mask = gold.ne(Constants.PAD)
                loss = -(one_hot * log_prb).sum(dim=1)
                loss = loss.masked_select(non_pad_mask).sum()  # average later
            else:
                loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')
            return loss

        ''' Apply label smoothing if needed '''
        loss = _cal_loss(pred, gold, smoothing)
        pred = pred.max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(Constants.PAD)
        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()
        return loss, n_correct


    def train(self, num_of_iters=1, data=None, hidden=None):
        self.loss = 0.0
        s = time.time()
        # zero the parameter gradients
        #self.optimizer.zero_grad()
        for i in range(num_of_iters):
            # if(not self.localsgd):
            #     logger.info('Adaptively adjust the learning rate.')
            self.adjust_learning_rate(self.train_epoch, self.optimizer)
            if self.train_iter % self.num_batches_per_epoch == 0 and self.train_iter > 0:
                self.train_epoch += 1
                logger.info('train iter: %d, num_batches_per_epoch: %d', self.train_iter, self.num_batches_per_epoch)
                #self.adjust_learning_rate(self.train_epoch, self.optimizer)
                logger.info('Epoch %d, avg train acc: %f, lr: %f, avg loss: %f' % (self.train_iter//self.num_batches_per_epoch, np.mean(self.train_acc_top1), self.lr, self.avg_loss_per_epoch/self.num_batches_per_epoch))
                #mean_s = np.mean(self.sparsities)
                #if self.train_iter>0 and np.isnan(mean_s):
                #    logger.warn('NaN detected! sparsities:  %s' % self.sparsities)
                    #sys.exit('NaN detected!!')
                #logger.info('Average Sparsity: %f, compression ratio: %f, communication size: %f', np.mean(self.sparsities), np.mean(self.compression_ratios), np.mean(self.communication_sizes))
                if self.rank == 0 and self.writer is not None:
                    self.writer.add_scalar('cross_entropy', self.avg_loss_per_epoch/self.num_batches_per_epoch, self.train_epoch)
                    self.writer.add_scalar('top-1_acc', np.mean(self.train_acc_top1), self.train_epoch)
                    #self.print_weight_gradient_ratio()
                #if self.rank == 0:
                #    with torch.no_grad():
                #        self.test(self.train_epoch)
                self.sparsities = []
                self.compression_ratios = []
                self.communication_sizes = []
                self.train_acc_top1 = []
                self.epochs_info.append(self.avg_loss_per_epoch/self.num_batches_per_epoch)
                # self.avg_loss_per_epoch = self.avg_loss_per_epoch/self.num_batches_per_epoch
                self.avg_loss_per_epoch = 0.0
                #self.data_iterator = iter(self.trainloader)
                #if self.train_iter > 0 and self.train_iter % 100 == 0:
                #    self.print_weight_gradient_ratio()
                # Save checkpoint
                if self.train_iter > 0 and self.rank == 0:
                    state = {'iter': self.train_iter, 'epoch': self.train_epoch, 'lr': self.lr, 'state': self.get_model_state()}
                    if self.prefix:
                        relative_path = './weights/%s/%s-n%d-bs%d-lr%.4f' % (self.prefix, self.dnn, self.nworkers, self.batch_size, self.base_lr)
                    else:
                        relative_path = './weights/%s-n%d-bs%d-lr%.4f' % (self.dnn, self.nworkers, self.batch_size, self.base_lr)
                    #utils.create_path(relative_path)
                    #filename = '%s-rank%d-epoch%d.pth'%(self.dnn, self.rank, self.train_epoch)
                    #fn = os.path.join(relative_path, filename)
                    #if self.train_epoch % 1 == 0:
                    #    self.save_checkpoint(state, fn)
                    #    self.remove_dict(state)
                if self.train_sampler and (self.nworkers > 1):
                    self.train_sampler.set_epoch(self.train_epoch)

            ss = time.time()
            if data is None:
                data = self.data_iter()

            if self.dataset == 'an4':
                inputs, labels_cpu, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            elif self.dataset == 'wmt2016':
                src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.cuda(), data)
                gold = tgt_seq[:, 1:]
            else:
                inputs, labels_cpu = data
            if self.is_cuda:
                if self.dnn in ['lstm', 'lstmwt2']:
                    inputs = Variable(inputs.transpose(0, 1).contiguous()).cuda()
                    labels = Variable(labels_cpu.transpose(0, 1).contiguous()).cuda()
                elif self.dnn != 'transformer':
                    inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
            else:
                labels = labels_cpu
                
            # wrap them in Variable
            #inputs, labels = Variable(inputs), Variable(labels)
            #logger.info('[%d] labels: %s', self.train_iter, labels_cpu)
            self.iotime += (time.time() - ss)
            
            sforward = time.time()
            if self.dnn == 'lstman4':
                out, output_sizes = self.net(inputs, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                loss = self.criterion(out, labels_cpu, output_sizes, target_sizes)
                self.forwardtime += (time.time() - sforward)
                loss = loss / inputs.size(0)  # average the loss by minibatch
            elif self.dnn in ['lstm', 'lstmwt2']:
                hidden = lstmpy.repackage_hidden(hidden)
                #print(inputs.size(), hidden[0].size(), hidden[1].size())
                outputs, hidden = self.net(inputs, hidden)
                tt = torch.squeeze(labels.view(-1, self.net.batch_size * self.net.num_steps))
                loss = self.criterion(outputs.view(-1, self.net.vocab_size), tt)
                self.forwardtime += (time.time() - sforward)
            elif self.dnn == 'transformer':
                pred = self.net(src_seq, src_pos, tgt_seq, tgt_pos)

                loss, n_correct = self.cal_performance_transformer(pred, gold, smoothing=True)
                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()
                accuracy = n_correct/n_word
                self.train_acc_top1.append(accuracy)
                self.forwardtime += (time.time() - sforward)
            else:
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                self.forwardtime += (time.time() - sforward)
            sbackward = time.time()
            if self.amp_handle is not None:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                    loss = scaled_loss
            else:
                loss.backward()
            loss_value = loss.item()
            self.backwardtime += (time.time() - sbackward)
            self.backwardtime_tmp = time.time() - sbackward
            # logger.info statistics
            self.loss += loss_value 

            self.avg_loss_per_epoch += loss_value

            if self.dnn not in ['lstm', 'lstmwt2', 'lstman4', 'transformer']:
                acc1, = self.cal_accuracy(outputs, labels, topk=(1,))
                self.train_acc_top1.append(float(acc1))
                
            self.train_iter += 1
        self.num_of_updates_during_comm += 1
        self.loss /= num_of_iters 
        self.timer += time.time() - s 
        display = 40
        if self.train_iter % display == 0:
            logger.warn('[%3d][%5d/%5d][rank:%d] loss: %.3f, average forward (%f) and backward (%f) time: %f, iotime: %f ' %
                  (self.train_epoch, self.train_iter, self.num_batches_per_epoch, self.rank,  self.loss, self.forwardtime/display, self.backwardtime/display, self.timer/display, self.iotime/display))
            self.timer = 0.0
            self.iotime = 0.0
            self.forwardtime = 0.0
            self.backwardtime = 0.0
            #if len(self.delays) > 0:
            #    delay = int(np.mean(self.delays))
            #else:
            #    delay = 0
            #logger.info('Delay interval: %d, average delay: %d', self.num_of_updates_during_comm- self.average_iter, delay)
            #self.delays = []
            #if self.is_cuda:
            #    torch.cuda.empty_cache()
            #self.print_weight_gradient_ratio()
            
        if self.dnn in ['lstm', 'lstmwt2']:
            return num_of_iters, hidden
        return num_of_iters

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct_top1 = 0
        correct_top5 = 0
        top1_acc = []
        top5_acc = []
        total = 0
        total_steps = 0
        costs = 0.0
        total_iters = 0
        total_wer = 0
        for batch_idx, data in enumerate(self.testloader):

            if self.dataset == 'an4':
                inputs, labels_cpu, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            else:
                inputs, labels_cpu = data
            if self.is_cuda:
                if self.dnn in ['lstm', 'lstmwt2']:
                    inputs = Variable(inputs.transpose(0, 1).contiguous()).cuda()
                    labels = Variable(labels_cpu.transpose(0, 1).contiguous()).cuda()
                else:
                    inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
            else:
                labels = labels_cpu

            if self.dnn in ['lstm', 'lstmwt2']:
                hidden = self.net.init_hidden()
                hidden = lstmpy.repackage_hidden(hidden)
                #print(inputs.size(), hidden[0].size(), hidden[1].size())
                outputs, hidden = self.net(inputs, hidden)
                tt = torch.squeeze(labels.view(-1, self.net.batch_size * self.net.num_steps))
                loss = self.criterion(outputs.view(-1, self.net.vocab_size), tt)
                test_loss += loss.item()
                costs += loss.item() * self.net.num_steps
                total_steps += self.net.num_steps
            elif self.dnn == 'lstman4':
                targets = labels_cpu
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                out, output_sizes = self.net(inputs, input_sizes)
                decoded_output, _ = self.decoder.decode(out.data, output_sizes)

                target_strings = self.decoder.convert_to_strings(split_targets)

                wer, cer = 0, 0
                target_strings = self.decoder.convert_to_strings(split_targets)
                wer, cer = 0, 0
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    wer += self.decoder.wer(transcript, reference) / float(len(reference.split()))
                total_wer += wer

            else:
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                acc1, acc5 = self.cal_accuracy(outputs, labels, topk=(1, 5))
                batch_size = labels.size(0)
                correct_top1 += float(acc1) * batch_size
                correct_top5 += float(acc5) * batch_size

                #top1_acc.append(float(acc1))
                #top5_acc.append(float(acc5))

                test_loss += loss.data.item()
                #_, predicted = torch.max(outputs.data, 1)
                #correct += predicted.eq(labels.data).cpu().sum()
            total += labels.size(0)
            total_iters += 1
        test_loss /= total_iters
        if self.dnn not in ['lstm', 'lstmwt2', 'lstman4']:
            acc = correct_top1/total
            acc5 = correct_top5/total
            #acc = np.mean(top1_acc)
            #acc5 = np.mean(top5_acc)
        elif self.dnn in ['lstm', 'lstmwt2']:
            acc = np.exp(costs / total_steps)
            acc5 = 0.0
        elif self.dnn == 'lstman4':
            wer = total_wer / len(self.testloader.dataset)
            acc = wer
            acc5 = 0.0
        loss = float(test_loss)/total
        logger.info('Epoch %d, lr: %f, val loss: %f, val top-1 acc: %f, top-5 acc: %f' % (epoch, self.lr, test_loss, acc, acc5))
        self.net.train()
        return acc

    def update_model(self):
        self.optimizer.step()
        if settings.EXCHANGE_MODE == 'MODEL+GRAD':
            for name, parameter in self.net.named_parameters():
                #phok = np.sum([self.m ** i for i in range(1,self.train_iter-self.average_iter+1)])
                if name not in self.v:
                    self.v[name] = parameter.grad.clone()
                else:
                    self.v[name] = self.m * self.v[name] + parameter.grad

    def encode_param(self, param, name=None):
        if not settings.SPARSE:
            param = param.cpu().numpy()
            return param.tobytes()
        elif not name.find('weight') >= 0: # only compress weights
            param = param.cpu().numpy()
            return param.tobytes()
        #    if settings.SPARSE:
        #        return huffman.fullencode(param.tobytes())
        #    else:
        #        return param.tobytes()
        #alpha = 0.95; beta = 0.1
        #nnz = np.count_nonzero(param==0.)
        #if name in self.remainer:
        #    pass
        #    #residuals = self.remainer[name] != 0.
        #    #param[residuals] = (param[residuals] + self.remainer[name][residuals])/2.
        #    #param[residuals] = alpha*param[residuals] + (1-alpha)*self.remainer[name][residuals]
        #    #param[residuals] = param[residuals] + self.remainer[name][residuals]
        #    #self.remainer[name][residuals] = param[residuals] - self.remainer[name][residuals]
        #else:
        #    self.remainer[name] = np.zeros(param.shape, dtype=param.dtype)
        #logger.debug('nnz before: %d, size: %d', nnz, param.size)
        #vals = param.flatten()
        #mean = np.mean(vals)
        #std = np.std(vals)
        #num_epoch = self.train_iter / self.num_batches_per_epoch 
        #if num_epoch >= len(self.target_sparsities):
        #    num_epoch = len(self.target_sparsities) - 1 
        #s = self.target_sparsities[num_epoch]
        #thres = s*2*std 
        #zero_condition = np.abs(param-mean) < thres
        #self.remainer[name][zero_condition] = self.remainer[name][zero_condition] * beta + param[zero_condition]
        #self.remainer[name][zero_condition] += param[zero_condition]
        #param[zero_condition] = 0.
        #sampling = np.random.randint(2, size=param[zero_condition].size)
        try:
            if name not in self.distributions:
                if self.is_cuda:
                    z = torch.zeros(param.shape, dtype=torch.float32, device=torch.cuda.current_device())
                else:
                    z = torch.zeros(param.shape, dtype=torch.float32)
                sparsity = self.sparsity
                if name.find('fc') >= 0:
                    sparsity = 0.999
                p = 1-sparsity
                z += p
                self.distributions[name] = z
            else:
                z = self.distributions[name]
            sampling = torch.bernoulli(z)
            zero_param = param * sampling
            param = zero_param.cpu().numpy()
            #sampling = np.random.binomial(1, 1-0.9, size=param.size)
            #sampling = sampling.reshape(param.shape)
            #param *= sampling
        except Exception as e:
            logger.error('Exception: %s', e)
        #mean_of_zeros = np.mean(param[zero_condition])
        #std_of_zeros = np.std(param[zero_condition])
        #param[zero_condition] = 0. #np.random.normal(mean_of_zeros, std_of_zeros, param[zero_condition].shape)
        #param[zero_condition] = np.sign(param[zero_condition]) * thres/2. 
        nnz = np.count_nonzero(param)
        real_s = (param.size-nnz)*1.0/param.size
        if np.isnan(real_s):
            logger.warn('NaN detected! nnz: %d, size: %d' % (nnz, param.size))
        self.sparsities.append(real_s)
        #logger.debug('nnz after: %d, sparsity: %f', nnz, nnz*1.0/param.size)
        dumps = param.tobytes()
        original_size = len(dumps) 
        if settings.SPARSE:
            #dumps = huffman.fullencode(dumps)
            dumps = huffman.encode_with_indexs(param)
            self.compression_ratios.append(original_size*1.0/len(dumps))
        return dumps

    def decode_param(self, data, name):
        if settings.SPARSE:
            if name.find('weight') >= 0:
                #data = huffman.fulldecode(data)
                if self.is_cuda and settings.GPU_CONSTRUCTION:
                    gpu_mem = self.gpu_caches.get(name, None)
                else:
                    gpu_mem = None
                data = huffman.decode_with_indexs(data, gpu_mem)
                if self.is_cuda and gpu_mem is None and settings.GPU_CONSTRUCTION:
                    data = torch.from_numpy(data)
                    self.gpu_caches[name] = data.cuda()
                elif not settings.GPU_CONSTRUCTION:
                    data = torch.from_numpy(data)
                return data
        dumps = data
        #arr = np.frombuffer(dumps, dtype=np.float16).astype(np.float32)
        arr = np.frombuffer(dumps, dtype=np.float32)
        arr = torch.from_numpy(arr)
        if self.is_cuda:
            arr = arr.cuda()
        return arr

    def encode_model(self, model):
        #total_size = 0
        s = time.time()
        serialized = []
        serialized.append(struct.pack('i', len(model.keys())))
        pciet = 0.
        for name, param in model.items():
            tmpt = time.time()
            #ny = param.cpu().numpy()
            ny = param
            pciet += tmpt-time.time()
            #ny = self.ternarize(param).cpu().numpy()
            #ny = param.cpu().numpy().astype(np.float16)
            byteparam = self.encode_param(ny, name)
            serialized.append(struct.pack('i', len(name)))
            #logger.debug('encode name l: %d', len(name))
            serialized.append(name)
            serialized.append(struct.pack('i', len(byteparam)))
            #logger.debug('encode model l: %d', len(byteparam))
            serialized.append(byteparam)
            #total_size += ny.size()
        #logger.debug('model total size: %d, --get model time used: %f, pcie time: %f', total_size * 4, time.time()-s, tmpt)
        serialized = b''.join(serialized)
        return serialized 

    def decode_model(self, serialized):
        own_state = {}
        offset = 0
        num_item = struct.unpack('i', serialized[offset:offset+4])[0]
        offset += 4
        for i in range(num_item):
            l = struct.unpack('i', serialized[offset:offset+4])[0]
            #logger.debug('decode name l: %d', l)
            offset += 4
            name = serialized[offset:offset+l]
            offset += l
            l = struct.unpack('i', serialized[offset:offset+4])[0]
            #logger.debug('decode model l: %d', l)
            offset += 4
            param = serialized[offset:offset+l]
            offset += l
            own_state[name] = param
        return own_state

    def _get_original_params(self, mode=settings.EXCHANGE_MODE):
        if mode == 'MODEL':
            own_state = self.net.state_dict()
            return own_state
        elif mode == 'GRAD':
            grad_of_params = {}
            for name, parameter in self.net.named_parameters():
                grad_of_params[name] = parameter.grad
            return grad_of_params
        elif mode == 'MODEL+GRAD':
            model_and_grad = {}
            for name, parameter in self.net.named_parameters():
                model_and_grad[name] = parameter.data
                model_and_grad[name+b'_gradient'] = parameter.grad
            return model_and_grad 

    def get_model(self, mode=settings.EXCHANGE_MODE):
        grad_of_params = self._get_original_params(mode)
        encoded_model = self.encode_model(grad_of_params)
        self.communication_sizes.append(len(encoded_model))
        return encoded_model

    def ternarize(self, params):
        """
        Paper: TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning, W. Wen et al., 2017
        """
        c = 2.5
        std = torch.std(params)
        params = torch.clamp(params, min=-c*torch.abs(std), max=c*torch.abs(std))
        st = torch.max(torch.abs(params))
        propabilities = torch.abs(params) / st
        distribution = torch.distributions.Bernoulli(propabilities)
        b = distribution.sample()
        tern = st * torch.sign(params)  * b 
        #tern = torch.sign(params)  * b
        logger.debug('Tern norm: %f', torch.norm(tern, 2))
        return tern

    def param_average(self, a, b, ratio, is_asked, v=None, name=None):
        """
        b should be a Tensor
        """
        if self.is_cuda:
            a_tensor = a
            if b.is_cuda:
                b_tensor = b
            else:
                b_tensor = b.cuda()
        else:
            a_tensor = a.cpu() 
            b_tensor = b
        #if is_asked:
        #new_param = (a_tensor+b_tensor.view(a_tensor.size()))/2.0
        b_tensor = b_tensor.view(a_tensor.size())
        if settings.SPARSE:
            condition = b_tensor == 0.
            if name.find('weight') >= 0:
                new_param = (a_tensor + torch.where(condition, a_tensor, b_tensor))/2.
            else:
                new_param = (a_tensor+b_tensor)/2. 
            #new_param[condition] = new_param[condition] - self.lr * self.net.named_parameters()[name].grad[condition]
            #for n, parameter in self.net.named_parameters():
            #    if name == n:
            #        new_param[condition] = new_param[condition] - 2*self.lr * parameter.grad[condition]
            #        break
            #new_param = (1+self.lr) * a_tensor + self.lr * b_tensor
            #new_param = (a_tensor + b_tensor)/2.
            # clamp
            #std = torch.std(new_param)
            #new_param[torch.abs(new_param)>=3*std] = 0.
            #new_param = (a_tensor + b_tensor)/2.
            #std = torch.std(new_param)
            #new_param[torch.abs(new_param)>=8*std] = torch.mean(new_param)
        else:
            if settings.EXCHANGE_MODE == 'MODEL+GRAD':
                v_tensor = torch.from_numpy(v).cuda()
                v_tensor = v_tensor.view(a_tensor.size())
                #new_param = ((b_tensor - self.lr * v_tensor) + a_tensor ) / 2. # Works
                #new_param = (b_tensor - self.lr * (v_tensor + 0.5*v_tensor * v_tensor * (a_tensor - b_tensor)) + a_tensor ) / 2. # Works
                #new_param = (b_tensor - self.lr * (v_tensor + 0.9*v_tensor * v_tensor * (a_tensor - b_tensor)) + a_tensor ) / 2.
                #interval = self.train_iter - self.average_iter
                #sump = 0.0
                #for j in range(1, interval+1):
                #    phok = float(np.sum([self.m ** i for i in range(1,j+1)]))
                #    sump+=phok
                #phok = float(np.sum([self.m ** i for i in range(1,interval+1)]))
                #square = v_tensor * v_tensor
                #g_dc = (v_tensor - square * b_tensor) * sump + square * self.v[name] + v_tensor * phok
                if name in self.v:
                    g_dc = self.v[name] * self.m + v_tensor
                else:
                    g_dc = v_tensor
                b_tensor_dc = b_tensor - self.lr * g_dc
                new_param = (a_tensor + b_tensor_dc) / 2.
                if name in self.v:
                    self.v[name][self.v[name]!=0.] = 0.
            else:
                new_param = (a_tensor + b_tensor)/2.
        del b_tensor
        del a_tensor
        return new_param

    def replace_model(self, model):
        recv_state = self.decode_model(model)
        own_state = self.net.state_dict()
        for name, param in own_state.items():
            if name in recv_state:
                a_tensor = own_state[name]
                b_tensor = self.decode_param(recv_state[name], name)
                b_tensor = b_tensor.view(a_tensor.size())
                own_state[name] = b_tensor 
        self.net.load_state_dict(own_state)

    def average_model(self, model, recved_loss, is_asked):
        #own_state = self.net.state_dict()
        #time.sleep(8)
        if settings.EXCHANGE_MODE == 'MODEL+GRAD':
            own_state = {}
            for name, parameter in self.net.named_parameters():
                own_state[name] = parameter.data
        else:
            own_state = self._get_original_params()
        s = time.time()
        loss = self.get_loss()
        average_ratio = 1.0
        if recved_loss > 0:
            r = (recved_loss - loss) / recved_loss
            if r > 0.2:
                #average_ratio = 1.0001#1+self.lr 
                average_ratio = 1.001#1+self.lr 
            elif r < -0.2:
                #average_ratio = 0.9999#1-self.lr 
                average_ratio = 0.999#1-self.lr 
            else:
                average_ratio = 1
        recv_state = self.decode_model(model)
        for name, param in own_state.items():
            #if name not in recv_state:
            #    continue
            remote_param = self.decode_param(recv_state[name], name)  
            v = None
            if settings.EXCHANGE_MODE == 'MODEL+GRAD':
                remote_gradients = self.decode_param(recv_state[name+b'_gradient'])
                #if settings.DELAY > 0:
                #    remote_param = remote_param-self.lr*remote_gradients * (self.train_iter-self.average_iter-1)
                #else:
                #remote_param = remote_param+0.25*self.lr*remote_gradients * (self.train_iter-self.average_iter)
                #remote_param = remote_param-self.lr*remote_gradients * (self.train_iter-self.average_iter-1)
                #correction = 0.0
                #for j in range(self.train_iter - self.average_iter+1):
                #    for k in range(j+1):
                #        correction += (1-0.9)**k

                #if name in self.v:
                #    v = 0.9 * self.v[name] + remote_gradients
                #    self.v[name] = v
                #else:
                #    v = remote_gradients
                #    self.v[name] = v
                v = remote_gradients

                #remote_param = remote_param - self.lr * v 
                #remote_param = remote_param - self.lr * v * v * (param - remote_param)
                #remote_param = remote_param-self.lr*remote_gradients*correction # momentum correction
            new_param = self.param_average(param, remote_param, average_ratio, is_asked, v, name) 
            del param
            del remote_param
            own_state[name] = new_param
        if settings.EXCHANGE_MODE == 'MODEL':
            self.net.load_state_dict(own_state)
        elif settings.EXCHANGE_MODE == 'GRAD':
            for name, parameter in self.net.named_parameters():
                parameter.grad.data = own_state[name].data
        elif settings.EXCHANGE_MODE == 'MODEL+GRAD':
            for name, parameter in self.net.named_parameters():
                parameter.data = own_state[name]
            #self.net.load_state_dict(own_state)
        logger.debug('====model average time: %f', time.time()-s)
        self.delays.append(self.num_of_updates_during_comm - self.average_iter)
        self.average_iter = self.num_of_updates_during_comm
        self.remove_dict(own_state)

    def update_with_remote_gradients(self, recved_gradients):
        recv_state = self.decode_model(recved_gradients)
        for name, parameter in self.net.named_parameters():
            recved = recv_state.get(name, None)
            if recved:
                b_tensor = self.decode_param(recved, name)
                b_tensor = b_tensor.view(parameter.size())
                parameter.grad = b_tensor 
        self.update_model()

    def remove_dict(self, dictionary):
        # keys = dictionary.keys()
        # for k in keys:
        #     del dictionary[k]
        dictionary.clear()

    def send_model(self, rank):
        own_state = self.net.state_dict()
        for name, param in own_state.items():
            dist.send(tensor=param, dst=rank)
        #logger.info("finished %s layer..." % name)

    def recv_model(self, rank):
        own_state = self.net.state_dict()
        for name, param in own_state.items():
            dist.recv(tensor=param, src=rank)
            own_state[name] = (own_state[name] + param) / 2.0
        #logger.info("finished %s layer..." % name)
        self.net.load_state_dict(own_state)

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)

    def _step(self, closure=None):
        """Performs a single optimization step.
            Arguments:
                closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
    
        for group in self.optimizer.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
    
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)
        return loss

    def zero_grad(self):
        self.optimizer.zero_grad()


def train_with_single(dnn, dataset, data_dir, nworkers, lr, batch_size, nsteps_update, max_epochs, num_steps=1):
    torch.cuda.set_device(0)
    trainer = DLTrainer(0, nworkers, dist=False, batch_size=batch_size, 
        is_weak_scaling=True, ngpus=1, data_dir=data_dir, dataset=dataset, 
        dnn=dnn, lr=lr, nworkers=nworkers, prefix='singlegpu', num_steps = num_steps)
    iters_per_epoch = trainer.num_batches_per_epoch #trainer.get_num_of_training_samples() // (nworkers * batch_size * nsteps_update)
    #seq_layernames, layerwise_times, layerwise_sizes = benchmark(trainer)
    #logger.info('Bencharmked backward time: %f', np.sum(layerwise_times))
    #logger.info('Model size: %d', np.sum(layerwise_sizes))
    norm_clip = None
    if dnn in ['lstm', 'lstmwt2']:
        norm_clip = 0.25
    elif dnn == 'lstman4':
        norm_clip = 400

    times = []
    display = 40 if iters_per_epoch > 40 else iters_per_epoch-1
    for epoch in range(max_epochs):
        if dnn in ['lstm', 'lstmwt2']:
            hidden = trainer.net.init_hidden()
        for i in range(iters_per_epoch):
            s = time.time()
            trainer.optimizer.zero_grad()
            for j in range(nsteps_update):
                if dnn in ['lstm', 'lstmwt2']:
                    _, hidden = trainer.train(1, hidden=hidden)
                else:
                    trainer.train(1)
            if norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainer.net.parameters(), norm_clip)

            trainer.update_model()
            times.append(time.time()-s)
            if i % display == 0 and i > 0: 
                time_per_iter = np.mean(times)
                logger.info('Time per iteration including communication: %f. Speed: %f images/s', time_per_iter, batch_size * nsteps_update / time_per_iter)
                times = []
            #if i > 1000:
            #    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single trainer")
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--nsteps-update', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='imagenet', choices=_support_datasets, help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=_support_dnns, help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--max-epochs', type=int, default=settings.MAX_EPOCHS, help='Default maximum epochs to train')
    parser.add_argument('--num-steps', type=int, default=35)
    args = parser.parse_args()
    batch_size = args.batch_size * args.nsteps_update
    prefix = settings.PREFIX
    relative_path = './logs/singlegpu-%s/%s-n%d-bs%d-lr%.4f-ns%d' % (prefix, args.dnn, 1, batch_size, args.lr, args.nsteps_update)
    utils.create_path(relative_path)
    logfile = os.path.join(relative_path, settings.hostname+'.log')
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)
    train_with_single(args.dnn, args.dataset, args.data_dir, 1, args.lr, args.batch_size, args.nsteps_update, args.max_epochs, args.num_steps)

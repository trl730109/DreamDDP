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

from transformers import (BertConfig, 
                          GPT2Config, 
                          BertForSequenceClassification, 
                          GPT2LMHeadModel, 
                          Trainer, 
                          TrainingArguments, 
                          DataCollatorForLanguageModeling, 
                          DataCollatorWithPadding,
                          CONFIG_MAPPING,
                          MODEL_MAPPING,
                          AutoConfig,
                          AutoModelForCausalLM,
                          AutoTokenizer,
                          SchedulerType,
                          default_data_collator,
                          get_scheduler,)
# from transformers import BertTokenizer, GPT2Tokenizer
from datasets import load_dataset, load_from_disk

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

_support_datasets = ['imagenet', 'cifar10', 'an4', 'ptb', 'wt2', 'mnist', 'wmt2016', 'shakespeare', 'wikitext2']
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
        'transformer',
        'gpt2', 'bert']


gpt_path = "/home/yinyiming/DDP-Train/gpt2/gpt2"
shakespeare_path = "/home/esetstore/dataset/shakespeare"
wikitext_path = '/home/yinyiming/datasets/wikitext2'

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


def get_available_gpu_device_ids(ngpus):
    return range(0, ngpus)

def create_net(dnn='gpt2', **kwargs):
    ext = None
    if dnn == 'gpt2':
        config = GPT2Config.from_pretrained(gpt_path)
        net = AutoModelForCausalLM.from_pretrained(
            gpt_path,
            from_tf=False, 
            config=config,
            low_cpu_mem_usage=True, 
            trust_remote_code=False
        )
    elif dnn == 'bert':
        pass
    else:
        errstr = 'Unsupport neural network %s' % dnn
        logger.error(errstr)
        raise errstr 
    return net, ext


class LLMTrainer:

    def __init__(self, rank, size, master='gpu10', localsgd=False, dist=True, ngpus=1, batch_size=32, 
        is_weak_scaling=True, data_dir='./data', dataset='wikitext2', dnn='gpt2', 
        lr=0.04, nworkers=1, prefix=None, sparsity=0.95, pretrain=None, num_steps=35, tb_writer=None, amp_handle=None,optimizer_name='Adam'):

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
        # if self.dataset == 'cifar10' or self.dataset == 'mnist':
        #     self.num_classes = 10
        # elif self.dataset == 'imagenet':
        #     self.num_classes = 1000
        # elif self.dataset == 'an4':
        #     self.num_classes = 29 
        # elif self.dataset in ['ptb', 'wt2']:
        #     self.num_classes = 10
        # elif self.dataset == 'wmt2016':
        #     self.num_classes = 10
        self.nworkers = nworkers # just for easy comparison
        self.data_dir = data_dir
        if type(dnn) != str:
            self.net = dnn
            self.dnn = dnn.name
            self.ext = None # leave for further parameters
        else:
            self.dnn = dnn
            # TODO: Refact these codes!
            if self.dnn == 'gpt2':
                if data_dir is not None:
                    self.data_prepare()
                self.net, self.ext = create_net(dnn='gpt2')
            elif self.dnn == 'bert':
                pass
  
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

    def wikitext2_prepare(self):
        # Data loading code
        tokenizer = AutoTokenizer.from_pretrained(gpt_path)
        dataset = load_from_disk(wikitext_path)
        
        def tokenize(example):
            return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

        # Apply the encoding to the dataset
        encoded_dataset = dataset.map(tokenize, batched=True)
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        trainset = encoded_dataset['train']
        valset = encoded_dataset['validation']
        self.trainset = trainset

        train_sampler = None
        shuffle = True
        if self.nworkers > 1: 
            # if settings.EFFICIENT_IO:
            #     train_sampler = CachedSampler(self.trainset, num_replicas=self.nworkers, 
            #             rank=self.rank, cached_index_images=self.cached_index_images)
            # else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=True, sampler=train_sampler)
        

        self.testset = valset
        self.testloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=8, pin_memory=True)

    def data_prepare(self):
        if self.dataset == 'wikitext2':
            self.wikitext2_prepare()
        elif self.dataset == 'shakespeare':
            pass
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
        for i in range(num_of_iters):
            self.adjust_learning_rate(self.train_epoch, self.optimizer)
            if self.train_iter % self.num_batches_per_epoch == 0 and self.train_iter > 0:
                self.train_epoch += 1
                logger.info('Epoch %d, avg train acc: %f, lr: %f, avg loss: %f' % (
                    self.train_iter // self.num_batches_per_epoch, 
                    np.mean(self.train_acc_top1), self.lr, self.avg_loss_per_epoch / self.num_batches_per_epoch
                ))
                if self.rank == 0 and self.writer is not None:
                    self.writer.add_scalar('cross_entropy', self.avg_loss_per_epoch / self.num_batches_per_epoch, self.train_epoch)
                    self.writer.add_scalar('top-1_acc', np.mean(self.train_acc_top1), self.train_epoch)
                self.sparsities = []
                self.compression_ratios = []
                self.communication_sizes = []
                self.train_acc_top1 = []
                self.epochs_info.append(self.avg_loss_per_epoch / self.num_batches_per_epoch)
                self.avg_loss_per_epoch = 0.0
                if self.train_sampler and (self.nworkers > 1):
                    self.train_sampler.set_epoch(self.train_epoch)

            ss = time.time()
            data = self.data_iter()

            inputs, labels_cpu = data
            if self.is_cuda:
                inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
            else:
                labels = labels_cpu
                    
            self.iotime += (time.time() - ss)
                
            sforward = time.time()
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
            
            self.loss += loss_value 
            self.avg_loss_per_epoch += loss_value

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

        self.update_model()
        return num_of_iters

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        total_iters = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(self.testloader):
                inputs, labels_cpu = data
                if self.is_cuda:
                    inputs, labels = inputs.cuda(non_blocking=True), labels_cpu.cuda(non_blocking=True)
                else:
                    labels = labels_cpu

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                acc1, acc5 = self.cal_accuracy(outputs, labels, topk=(1, 5))
                batch_size = labels.size(0)
                correct_top1 += float(acc1) * batch_size
                correct_top5 += float(acc5) * batch_size

                test_loss += loss.data.item()
                total += labels.size(0)
                total_iters += 1

        test_loss /= total_iters
        acc = correct_top1 / total
        acc5 = correct_top5 / total
        logger.info('Epoch %d, lr: %f, val loss: %f, val top-1 acc: %f, top-5 acc: %f' % (epoch, self.lr, test_loss, acc, acc5))
        
        self.net.train()
        return acc

    def update_model(self):
        self.optimizer.step()

    def encode_param(self, param, name=None):
        if not settings.SPARSE:
            param = param.cpu().numpy()
            return param.tobytes()
        elif not name.find('weight') >= 0: # only compress weights
            param = param.cpu().numpy()
            return param.tobytes()

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

        except Exception as e:
            logger.error('Exception: %s', e)

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
            nesterov = gro# -*- coding: utf-8 -*-

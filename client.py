# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import gc
import utils
import time
import threading
import argparse
import settings
import socket
import random
import logging
from torch.multiprocessing import Queue
from p2pnet import P2PNet
from csp2p.constants import ROLE
from dl_trainer import DLTrainer 
from settings import logger, formatter

gc.enable()

class ShareObject():
    host_ = None
    accuracy_ = 0.0

class SYNC_MODEL_STATUS:
    NONE = 'NONE'
    REQUSTED = 'requested'
    TRANSFERING = 'transfering'
    RECEIVED = 'received'


TIMEOUT = 10
DEFAULT_NUM_EXCHANGE_NODES=1
hostname = socket.gethostname()

class Client():

    def __init__(self, rank, max_epochs, world_size, bind, port, dnn, dataset, num_exchange_nodes=DEFAULT_NUM_EXCHANGE_NODES, ngpus=1, batch_size=32, role=ROLE.ACTIVE, data_dir='./data', nworkers=1, lr=0.1, sparsity=0.95, pretrain=None):
        #self.id_ = utils.gen_random_id() 
        self.dnn = dnn 
        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.sparsity = sparsity
        self.max_epochs = max_epochs
        self.nworkers = nworkers
        self.p2p_status = {}
        self.rank_ = rank
        self.world_size = world_size
        self.port = port
        self.net_ = P2PNet(self.rank_, bind=bind, port=port, callback=self, is_psworker=False)
        self.status_ = ShareObject()
        self.status_.host = self.net_.get_local_host()
        self.sync_model_status = SYNC_MODEL_STATUS.NONE
        self.ngpus = ngpus 
        self.update_freq = 1./ world_size 
        self.comm_timer = 0.0
        self.comm_counter = 0
        self.model_transfer_lock = threading.Lock()
        self.initiative_sync_status = SYNC_MODEL_STATUS.NONE
        self.passive_sync_status = SYNC_MODEL_STATUS.NONE
        self.num_exchange_nodes = num_exchange_nodes
        self.multiple_active_transfers = {}
        self.multiple_passive_transfers = {}
        self.role = role
        self.pretrain = pretrain 
        self.model_queue = Queue()

    def update_self_role(self):
        pass

    def run(self):
        self.p2pthread = threading.Thread(name='p2pnode', target=self.net_.run)
        self.p2pthread.start()

        self.trainer = DLTrainer(self.rank_, self.world_size, dist=False, ngpus=self.ngpus, batch_size=self.batch_size, is_weak_scaling=True, dataset=self.dataset, dnn=self.dnn, data_dir=self.data_dir, lr=self.lr, nworkers=self.nworkers, prefix=settings.PREFIX, sparsity=self.sparsity, pretrain=self.pretrain)
        max_iter = self.trainer.get_num_of_training_samples()/(self.batch_size * self.nworkers) * self.max_epochs
        self.max_iter = max_iter
        logger.info('train inited...')
        self.p2p_status[self.net_.get_local_host()] = self.status_
        i = 0 
        display = 100
        logger.info('Start to train...')
        s = time.time()
        while(1):
            ss = time.time()
            try:
                self.model_transfer_lock.acquire()
                self.trainer.zero_grad()
                num_iters = self.trainer.train(settings.DELAY_COMM)
                if self.model_transfer_lock.locked():
                    self.model_transfer_lock.release()
            except Exception as e:
                logger.error('Train exception: %s', e)
                break
            target_hosts = self.check_status()
            if len(target_hosts) > 0:
                self.ask_for_model(target_hosts)

            if i > 0 and self.role == ROLE.PASSIVE and settings.PASSIVE_WAIT:
                # Wait for active
                qitem = self.model_queue.get()

            logger.debug('Try to acquire train model lock')
            sst = time.time()
            self.model_transfer_lock.acquire()
            logger.debug('Continue train model lock acquired, time used: %s', time.time()-sst)

            self.comm_timer += time.time() - ss
            self.comm_counter += 1
            if i % display == 0 and i > 0:
                logger.info('Real batch time include communication: %f, role: %s, [%d <-- %s]' % (self.comm_timer/display, self.role, self.rank_, target_hosts))
                logger.info('Speed: %f images/s', self.batch_size / (self.comm_timer/display))
                self.comm_timer = 0.0
                self.comm_counter = 0
            try:
                self.trainer.update_model()
                if self.model_transfer_lock.locked():
                    self.model_transfer_lock.release()
            except Exception as e:
                logger.error('Model update exception: %s', e, exc_info=True)
                if self.model_transfer_lock.locked():
                    self.model_transfer_lock.release()
                break
            #if i % 5000 == 0 and self.role == ROLE.ACTIVE:
            #    logger.info('Save checkpoint at iteration %d', i)
            #    state = {'iter': i,'state': self.trainer.get_model_state()}
            #    filename = './weights/%s-%s-%d-iter%d.pth'%(self.dnn, hostname, self.port, i)
            #    self.trainer.save_checkpoint(state, filename)

            logger.debug('Train model lock released')
            logger.debug('=========Iteration time: %s', time.time()-ss)
            i += num_iters
            self.status_.accuracy_ = self.trainer.get_loss()
            self.net_.send_accuracy(self.status_.accuracy_, self.rank_, self.role, self.trainer.train_iter)
            if i >= self.max_iter:
                break
        self.net_.stop()
        logger.info('Train %d iterations, time used: %f' % (self.max_iter, time.time() - s))
        logger.info('All losses: %s', self.trainer.epochs_info)

    def ask_for_model(self, target_hosts):
        #self.model_fromrank = rank
        if not settings.ACTIVE_WAIT and self.initiative_sync_status != SYNC_MODEL_STATUS.NONE:
            return
        logger.debug('Try to acquire ask model lock from hosts: %s', target_hosts)
        logger.debug('Ask model lock acquired')
        self.initiative_sync_status = SYNC_MODEL_STATUS.REQUSTED
        for t in target_hosts:
            train_host = t[0]
            remote_rank = t[1]
            acc = t[2]
            #self.multiple_active_transfers[train_host] = SYNC_MODEL_STATUS.REQUSTED
            model_loss = self.get_local_model()
            self.net_.ask_model_from(remote_rank, train_host, model_loss)

    def check_status(self):
        target_hosts = []
        if self.role == ROLE.PASSIVE:
            return target_hosts
        accs = self.net_.get_accs()
        #print 'got accs: ', accs
        data = accs.get('data', None)
        exchange = accs.get('exchange', None)
        if not data or not exchange:
            return target_hosts 
        th = exchange.get(self.net_.get_local_host(), None)
        if not th:
            return target_hosts
        train_host = None
        remote_rank = -1
        acc = 100.0

        active_set = []
        passive_set = []
        for dk in data:
            d = data[dk]
            rrank = d['rank']
            train_host = d['train_host']
            role = d['role']
            acc = d['accuracy']
            if train_host == th:
                target_hosts.append([train_host, rrank, acc])
                return target_hosts
        return target_hosts
        #    if role == ROLE.ACTIVE:
        #        active_set.append(d)
        #    else:
        #        passive_set.append(d)
        #logger.debug('active_set: %s', active_set)
        #logger.debug('passive_set: %s', passive_set)
        #if len(passive_set) == 0:
        #    return target_hosts

        #candidate_ranks = []
        #candidate_idx = random.randint(0, len(passive_set)-1)
        #candidate_ranks.append(candidate_idx)

        #while len(passive_set) >= self.num_exchange_nodes and len(candidate_ranks) < self.num_exchange_nodes:
        #    candidate_idx = random.randint(0, len(passive_set)-1)
        #    if candidate_idx not in candidate_ranks:
        #        candidate_ranks.append(candidate_idx)

        #for i in candidate_ranks:
        #    d = passive_set[i]
        #    remote_rank = d['rank']
        #    train_host = d['train_host']
        #    acc = d['accuracy']
        #    target_hosts.append([train_host, remote_rank, acc])
        #return target_hosts

    def get_local_model(self):
        return self.trainer.get_model(), self.trainer.get_loss()

    def model_transfering(self, msg):
        #if msg['is_asked'] == 1:
        #if self.initiative_sync_status == SYNC_MODEL_STATUS.REQUSTED and msg['is_asked'] == 1:
        is_asked = msg['is_asked']
        logger.debug('model_transfering start delay')
        if settings.DELAY > 0:
            time.sleep(settings.DELAY)
        logger.debug('model_transfering end delay')
        if is_asked == 1:
            if not settings.ACTIVE_WAIT:
                self.model_transfer_lock.acquire()
            remote_train_host = msg['train_host']
            logger.debug('Initiative model recieved, current status: %s, train_host:%s', self.initiative_sync_status, msg['train_host'])
            #self.multiple_active_transfers[remote_train_host] = SYNC_MODEL_STATUS.TRANSFERING

            self.initiative_sync_status  = SYNC_MODEL_STATUS.TRANSFERING
            recved_model = msg['model']
            recved_loss = msg['loss']
            self.trainer.average_model(recved_model, recved_loss, is_asked)
            #self.trainer.update_model()
            self.initiative_sync_status = SYNC_MODEL_STATUS.RECEIVED
            self.multiple_active_transfers.pop(remote_train_host, None)
            #if len(self.multiple_active_transfers) == 0:
            if self.model_transfer_lock.locked():
                self.model_transfer_lock.release()
                logger.debug('Ask model lock released')
            self.initiative_sync_status = SYNC_MODEL_STATUS.NONE
        #if self.passive_sync_status == SYNC_MODEL_STATUS.NONE and msg['is_asked'] == 0:
        elif is_asked == 0:
            remote_train_host = msg['train_host']
            logger.debug('Passive model recieved, current status: %s, train_host: %s', self.passive_sync_status, msg['train_host'])
            self.model_transfer_lock.acquire()
            recved_model = msg['model']
            recved_loss = msg['loss']
            self.trainer.average_model(recved_model, recved_loss, is_asked)
            #self.multiple_passive_transfers.pop(remote_train_host, None)
            #self.trainer.update_model()
            #self.passive_sync_status = SYNC_MODEL_STATUS.RECEIVED
            #if len(self.multiple_passive_transfers) == 0:
            if self.model_transfer_lock.locked():
                self.model_transfer_lock.release()
            #    logger.debug('Passive model lock released')
            self.passive_sync_status = SYNC_MODEL_STATUS.NONE
            if settings.PASSIVE_WAIT:
                self.model_queue.put('MSG')

    def train_server_disconnected(self, p):
        if self.model_transfer_lock.locked():
            self.model_transfer_lock.release()
        if self.initiative_sync_status != SYNC_MODEL_STATUS.NONE:
            self.initiative_sync_status = SYNC_MODEL_STATUS.NONE
            if settings.PASSIVE_WAIT:
                self.model_queue.put('MSG')
        logger.debug('Ask model lock released')

    def handle_modelreq(self, p, msg):
        self.passive_sync_status = SYNC_MODEL_STATUS.TRANSFERING
        #self.model_queue.put(msg)
        #self.model_transfering(msg)
        #if self.passive_sync_status == SYNC_MODEL_STATUS.NONE:
        #    remote_train_host = msg['host']
        #    #self.passive_sync_status = SYNC_MODEL_STATUS.TRANSFERING
        #    if len(self.multiple_passive_transfers) == 0:
        #        logger.debug('Try to acquire passive model lock, from host: %s', msg['host'])
        #        self.model_transfer_lock.acquire()
        #        logger.debug('Passive model lock acquired')
        #    self.multiple_passive_transfers[remote_train_host] = SYNC_MODEL_STATUS.TRANSFERING

    def handle_change_role(self, msg):
        role = self.role
        self.role = ROLE.ACTIVE if self.role == ROLE.PASSIVE else ROLE.PASSIVE
        logger.info('Change role from %s to %s', role, self.role)

    def get_accs_reps(self, p, msg):
        data = msg.get('data', [])
        current_nworker = len(data.keys())
        if current_nworker != self.nworkers and self.trainer.train_iter > 1000:
            logger.info('Number of worker changes from %d to %d', self.nworkers, current_nworker)
            self.nworkers = current_nworker
            #try:
            #    self.model_transfer_lock.acquire()
            #    self.trainer.update_nworker(current_nworker)
            #    if self.model_transfer_lock.locked():
            #        self.model_transfer_lock.release()
            #except Exception as e:
            #    logger.error('Update nworker error: %s', e)


if __name__ == '__main__':
    # Test
    batch_size = 32
    parser = argparse.ArgumentParser(description="p2pdl")
    parser.add_argument('--bind', type=str, default=hostname)
    parser.add_argument('--port', type=int, default=settings.PORT)
    #parser.add_argument('--device-ids', type=str, default='-1')
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=1, help='Just for experiments, and it cannot be used in production')
    parser.add_argument('--num-exchange-nodes', type=int, default=1)
    parser.add_argument('--rank', type=int, help='Speciafy rank for synchonize algorithm')
    parser.add_argument('--role', type=str, default=ROLE.ACTIVE, choices=[ROLE.ACTIVE, ROLE.PASSIVE])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10', 'mnist'], help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=['resnet50', 'resnet20', 'vgg19', 'vgg16', 'mnistnet', 'alexnet'], help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    parser.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    parser.add_argument('--sparsity', type=float, default=0.95, help='Default sparsity for compressed communication')
    parser.add_argument('--pretrain', type=str, default=None, help='Load pretrained model')
    args = parser.parse_args()
    rank = args.rank
    batch_size = args.batch_size
    port = args.port
    if settings.PREFIX:
        relative_path = './logs/%s/%s-n%d-bs%d-lr%.4f' % (settings.PREFIX, args.dnn, args.nworkers, args.batch_size, args.lr)
    else:
        relative_path = './logs/%s-n%d-bs%d-lr%.4f' % (args.dnn, args.nworkers, args.batch_size, args.lr)
    if settings.SPARSE:
        relative_path += '-s%.5f' % args.sparsity
    utils.create_path(relative_path)

    logfile = os.path.join(relative_path, hostname+'-'+str(port)+'.log')

    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)
    num_of_workers = len(settings.BOOTSTRAP_LIST)
    max_epochs= settings.MAX_EPOCHS
    c = Client(rank, max_epochs, num_of_workers, bind=args.bind, port=port, dnn=args.dnn, dataset=args.dataset, num_exchange_nodes=args.num_exchange_nodes, ngpus=args.ngpu, batch_size=batch_size, role=args.role, data_dir=args.data_dir, nworkers=args.nworkers, lr=args.lr, sparsity=args.sparsity, pretrain=args.pretrain)
    c.run()

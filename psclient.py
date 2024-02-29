# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import utils
import time
import threading
import argparse
import settings
import socket
import random
import logging
import Queue
from p2pnet import P2PNet
from csp2p.constants import ROLE
from dl_trainer import DLTrainer 
from settings import logger, formatter


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

class PSClient():

    def __init__(self, rank, max_epochs, world_size, bind, port, dnn, dataset, num_exchange_nodes=DEFAULT_NUM_EXCHANGE_NODES, ngpus=1, batch_size=32, role=ROLE.ACTIVE, data_dir='./data', nworkers=1, lr=0.1, sparsity=0.95):
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
        self.net_ = P2PNet(self.rank_, bind=bind, port=port, callback=self, is_psworker=True)
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
        self.model_queue = Queue.Queue()
        self.target_hosts = [settings.SERVER_IP + ':' + str(settings.SERVER_PORT)]

    def update_self_role(self):
        pass

    def run(self):
        self.p2pthread = threading.Thread(name='p2pnode', target=self.net_.run)
        self.p2pthread.start()

        self.trainer = DLTrainer(self.rank_, self.world_size, dist=False, ngpus=self.ngpus, batch_size=self.batch_size, is_weak_scaling=True, dataset=self.dataset, dnn=self.dnn, data_dir=self.data_dir, lr=self.lr, nworkers=self.nworkers, prefix=settings.PREFIX, sparsity=self.sparsity)
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
            if i % display == 0:
                logger.info('Real batch time include communication: %f, role: %s, [%d <-- %s]' % (self.comm_timer/display, self.role, self.rank_, target_hosts))
                self.comm_timer = 0.0
                self.comm_counter = 0
            try:
                self.trainer.update_model()
                if self.model_transfer_lock.locked():
                    self.model_transfer_lock.release()
            except Exception as e:
                logger.error('Model update exception: %s', e)
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
            self.net_.send_accuracy(self.status_.accuracy_, self.rank_, self.role, i)
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
            train_host = t
            remote_rank = 0
            model_loss = self.get_local_model()
            self.net_.ask_model_from(remote_rank, train_host, model_loss)

    def check_status(self):
        return self.target_hosts

    def get_local_model(self):
        return self.trainer.get_model('GRAD'), self.trainer.get_loss()

    def model_transfering(self, msg):
        recved_model = msg['model']
        logger.debug('model_transfering start delay')
        if settings.DELAY > 0:
            time.sleep(settings.DELAY)
        logger.debug('model_transfering end delay')
        self.trainer.replace_model(recved_model)
        self.initiative_sync_status = SYNC_MODEL_STATUS.NONE
        if self.model_transfer_lock.locked():
            self.model_transfer_lock.release()

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
    c = PSClient(rank, max_epochs, num_of_workers, bind=args.bind, port=port, dnn=args.dnn, dataset=args.dataset, num_exchange_nodes=args.num_exchange_nodes, ngpus=args.ngpu, batch_size=batch_size, role=args.role, data_dir=args.data_dir, nworkers=args.nworkers, lr=args.lr, sparsity=args.sparsity)
    c.run()

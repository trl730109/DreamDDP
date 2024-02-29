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
#from multiprocessing import Process, Queue
import torch.multiprocessing as mp
from torch.multiprocessing import Process, SimpleQueue, Queue
from p2pnet import P2PNet
from csp2p.constants import ROLE, PROCESS_MSG_TYPE
from csp2p import constants
from dl_trainer import DLTrainer, ModelEncoder, create_net
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

msg_from_socket_queue = Queue() #SimpleQueue()
model_from_socket_queue = Queue() #SimpleQueue()
msg_to_socket_queue = Queue() #SimpleQueue()
model_to_socket_queue = Queue() #SimpleQueue()
mplock = mp.Lock()

class ClientProcess(Process):
#class ClientProcess:
    def __init__(self, rank, max_epochs, world_size, bind, port, dnn, dataset, num_exchange_nodes=DEFAULT_NUM_EXCHANGE_NODES, ngpus=1, batch_size=32, role=ROLE.ACTIVE, data_dir='./data', lock=None):
        Process.__init__(self)
        self.dnn = dnn 
        self.dataset = dataset
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.rank_ = rank
        self.world_size = world_size
        self.sync_model_status = SYNC_MODEL_STATUS.NONE
        self.ngpus = ngpus 
        self.update_freq = 1./ world_size 
        self.comm_timer = 0.0
        self.comm_counter = 0
        self.num_exchange_nodes = num_exchange_nodes
        self.multiple_active_transfers = {}
        self.multiple_passive_transfers = {}
        self.role = role
        self.msg_from_socket_queue = msg_from_socket_queue
        self.model_from_socket_queue = model_from_socket_queue 
        self.msg_to_socket_queue = msg_to_socket_queue 
        self.model_to_socket_queue = model_to_socket_queue 
        self.ip = bind
        self.port = port
        self.first = True
        self.connected_hosts = {}
        self.acc_dict = {}
        self.net = create_net(dnn)
        self.lock = lock

    def get_local_host(self):
        host = self.ip + ':' + str(self.port)
        return host

    def send_accuracy(self, acc, rank, role, iter):
        msg = {'t': PROCESS_MSG_TYPE.SEND_ACC}
        msg['d'] = (acc, rank, role, iter)
        self.msg_to_socket_queue.put(msg)
        #logger.debug('len of msg_to_socket_queue: %d', self.msg_to_socket_queue.qsize())

    def send_model_req(self, target_hosts):
        logger.debug('Send model req to socket process')
        msg = {'t': PROCESS_MSG_TYPE.SEND_MODELREQ}
        msg['d'] =  {'ths': target_hosts,
                'loss': self.trainer.get_loss()}
                #'ml': self.get_local_model()}
        self.msg_to_socket_queue.put(msg)

    def update_acc_thread(self):
        logger.info('Start thread to recieve accuracy')
        while self.is_running:
            msg = self.msg_from_socket_queue.get()
            if msg['t'] == PROCESS_MSG_TYPE.RECV_ACC:
                self.acc_dict = msg['d']
            elif msg['t'] == PROCESS_MSG_TYPE.GET_MODEL:
                #m = self.get_local_model()
                #self.model_to_socket_queue.put(m)
                loss = self.trainer.get_loss()
                self.model_to_socket_queue.put(loss)
            elif msg['t'] == PROCESS_MSG_TYPE.CONNECTED:
                host = msg['d']
                self.connected_hosts[host] = True
            elif msg['t'] == PROCESS_MSG_TYPE.EXIT:
                break
        self.is_running = False

    def start_update_acc_thread(self):
        self.update_thread = threading.Thread(name='update_acc_thread', target=self.update_acc_thread)
        self.update_thread.start()

    def run(self):
        self.is_running = True
        self.start_update_acc_thread()

        self.trainer = DLTrainer(self.rank_, self.world_size, dist=False, ngpus=self.ngpus, batch_size=self.batch_size, is_weak_scaling=True, dataset=self.dataset, dnn=self.net, data_dir=self.data_dir)
        max_iter = self.trainer.get_num_of_training_samples()/self.batch_size * self.max_epochs
        self.max_iter = max_iter
        logger.info('train inited...')
        i = 0 
        display = 100
        logger.info('Start to train...')
        s = time.time()
        while self.is_running:
            ss = time.time()
            try:
                num_iters = self.trainer.train()
                logger.debug('Forward and backward time: %f', time.time()-ss)
            except Exception as e:
                logger.error('Train exception: %s', e)
                break
            target_hosts = self.check_status()
            if len(target_hosts) > 0:
                s1 = time.time()
                self.send_model_req(target_hosts)
                msg = self.model_from_socket_queue.get()
                logger.debug('Communication time: %f', time.time()-s1)
                s2 = time.time()
                #self.model_transfering(msg)
                logger.debug('Model average time: %f', time.time()-s2)
            else: # passive
                self.lock.acquire()
                if not self.model_from_socket_queue.empty():
                    msg = self.model_from_socket_queue.get()
                    #self.model_transfering(msg)
                else:
                    logger.debug('No peers, continue to train')
                self.lock.release()
            self.trainer.update_model()
            self.comm_timer += time.time() - ss
            if i % display == 0:
                logger.info('Real batch time include communication: %f, role: %s, [%d <-- %s]' % (self.comm_timer/display, self.role, self.rank_, target_hosts))
                self.comm_timer = 0.0

            logger.debug('=========Iteration time: %s', time.time()-ss)
            i += num_iters
            accuracy = self.trainer.get_loss()
            self.send_accuracy(accuracy, self.rank_, self.role, i)
            if i >= self.max_iter:
                break
        self.is_running = False
        logger.info('Train %d iterations, time used: %f' % (i, time.time() - s))

    def check_status(self):
        target_hosts = []
        if self.role == ROLE.PASSIVE:
            return target_hosts
        accs = self.acc_dict
        data = accs.get('data', None)
        exchange = accs.get('exchange', None)
        if not data or not exchange:
            return target_hosts 
        th = exchange.get(self.get_local_host(), None)
        if not th:
            return target_hosts
        train_host = None
        acc = 100.0
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

    def get_trainer_net(self):
        return self.net

    def get_local_model(self):
        return self.trainer.get_model(), self.trainer.get_loss()

    def model_transfering(self, msg):
        is_asked = msg['is_asked']
        if is_asked == 1:
            logger.debug('Initiative model recieved, from train_host:%s', msg['train_host'])
            recved_model = msg['model']
            recved_loss = msg['loss']
            self.trainer.average_model(recved_model, recved_loss, is_asked)
        elif is_asked == 0:
            logger.debug('Passive model recieved, from train_host: %s', msg['train_host'])
            recved_model = msg['model']
            recved_loss = msg['loss']
            self.trainer.average_model(recved_model, recved_loss, is_asked)


class P2PNetProcess(Process):
    def __init__(self, rank, bind, port=settings.PORT, callback=None, model=None, ngpus=0, lock=None):
        Process.__init__(self)
        self.msg_from_socket_queue = msg_from_socket_queue
        self.msg_to_socket_queue = msg_to_socket_queue
        self.model_from_socket_queue = model_from_socket_queue
        self.model_to_socket_queue = model_to_socket_queue
        self.p2pnet = P2PNet(rank, bind=bind, port=port, callback=self)
        self.is_running = True
        self.model = model
        self.model_encoder = ModelEncoder(model, ngpus>0)
        self.lock = lock
        self.model_transfer_lock = None

    def monitor_thread(self):
        logger.info('Monitor thread started')
        while self.is_running:
            #logger.info('Running to recv process msg....')
            msg = self.msg_to_socket_queue.get()
            #logger.debug('Recieved msg')
            if msg['t'] == PROCESS_MSG_TYPE.SEND_ACC:
                d = msg['d']
                self.p2pnet.send_accuracy(d[0], d[1], d[2], d[3])
            elif msg['t'] == PROCESS_MSG_TYPE.SEND_MODELREQ:
                logger.debug('Receive model req from training process')
                target_hosts = msg['d']['ths']
                model_loss = self.model_encoder.encode(), msg['d']['loss']
                for t in target_hosts:
                    train_host = t[0]
                    remote_rank = t[1]
                    #acc = t[2]
                    logger.debug('Send model req to passive worker')
                    self.p2pnet.ask_model_from(remote_rank, train_host, model_loss)

    def run(self):
        logger.info('P2PNetProcess started')
        self.mt = threading.Thread(name='p2pnode', target=self.monitor_thread)
        self.mt.start()
        self.p2pnet.run()

    def train_server_disconnected(self, p):
        pass

    def handle_change_role(self, recv_msg):
        pass

    def handle_modelreq(self, p, recv_msg):
        #self.model_from_socket_queue.put(recv_msg)
        pass

    #def model_transfering(self, recv_msg):
    #    logger.debug('Recv model from target worker')
    #    self.model_from_socket_queue.put(recv_msg)

    def get_local_model(self):
        m = {}
        m['t'] = PROCESS_MSG_TYPE.GET_MODEL
        self.msg_from_socket_queue.put(m)
        #model, loss = self.model_to_socket_queue.get()
        loss = self.model_to_socket_queue.get()
        model = self.model_encoder.get_model()
        return model, loss

    def get_accs_reps(self, p, msg):
        m = {}
        m['t'] = PROCESS_MSG_TYPE.RECV_ACC
        m['d'] = msg
        self.msg_from_socket_queue.put(m)

    def model_transfering(self, msg):
        is_asked = msg['is_asked']
        if is_asked == 1:
            logger.debug('Initiative model recieved, from train_host:%s', msg['train_host'])
            recved_model = msg['model']
            recved_loss = msg['loss']
            self.model_encoder.average_model(recved_model, recved_loss, is_asked)
            self.model_from_socket_queue.put(msg)
        elif is_asked == 0:
            logger.debug('Passive model recieved, from train_host: %s', msg['train_host'])
            recved_model = msg['model']
            recved_loss = msg['loss']
            self.lock.acquire()
            self.model_encoder.average_model(recved_model, recved_loss, is_asked)
            self.model_from_socket_queue.put(msg)
            self.lock.release()



def main():
    # Test
    batch_size = 32
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser(description="p2pdl")
    parser.add_argument('--bind', type=str, default=hostname)
    parser.add_argument('--port', type=int, default=settings.PORT)
    #parser.add_argument('--device-ids', type=str, default='-1')
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--num-exchange-nodes', type=int, default=1)
    parser.add_argument('--rank', type=int, help='Speciafy rank for synchonize algorithm')
    parser.add_argument('--role', type=str, default=ROLE.ACTIVE, choices=[ROLE.ACTIVE, ROLE.PASSIVE])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10'], help='Specify the dataset for training')
    parser.add_argument('--dnn', type=str, default='resnet50', choices=['resnet50', 'resnet20'], help='Specify the neural network for training')
    parser.add_argument('--data-dir', type=str, default='./data', help='Specify the data root path')
    args = parser.parse_args()
    rank = args.rank
    batch_size = args.batch_size
    port = args.port

    hdlr = logging.FileHandler(hostname+'-'+str(port)+'.log')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)


    num_of_workers = len(settings.BOOTSTRAP_LIST)
    max_epochs= settings.MAX_EPOCHS
    c = ClientProcess(rank, max_epochs, num_of_workers, bind=args.bind, port=port, dnn=args.dnn, dataset=args.dataset, num_exchange_nodes=args.num_exchange_nodes, ngpus=args.ngpu, batch_size=batch_size, role=args.role, data_dir=args.data_dir, lock=mplock)
    #c.run()
    c.start()

    p = P2PNetProcess(rank, bind=args.bind, port=port, model=c.get_trainer_net(), ngpus=args.ngpu, lock=mplock)
    p.start()

    p.join()
    c.join()

if __name__ == '__main__':
    main()

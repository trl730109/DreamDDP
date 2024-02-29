# -*- coding: utf-8 -*-
from __future__ import print_function
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.internet.protocol import Protocol, Factory
from twisted.internet import reactor
from threading import Lock, Thread
#import Queue
import sys
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue
else:
    import queue as Queue
from uuid import uuid4
#import simplejson as json
import ujson as json
import pickle
import struct
import time
import numpy as np
import argparse
from datetime import datetime
import logging
import settings
import torch.optim as optim
from settings import logger, formatter

import csp2p.constants as const
from random import shuffle
from csp2p.constants import ROLE
#from . import constants as const
from ctypes import create_string_buffer
from dl_trainer import DLTrainer

generate_nodeid = lambda: str(uuid4())[0:8]

class FastBufferString:
    def __init__(self, size=1024*1024*1024):
        self.size = size
        #self.buf = create_string_buffer(size) 
        self.buf = []#create_string_buffer(size) 
        self.offset = 0

    def insert(self, string):
        self.s = time.time()
        length = len(string)
        #self.buf[self.offset:self.offset+length] = string 
        self.buf.append(string)
        self.offset += length
        #logger.info('Insert time: ', time.time() - self.s)

    def clear(self):
        self.buf = []
        self.s = time.time()
        self.offset = 0

    def get(self):
        #logger.info('cat time used: ', time.time() - self.s)
        #return self.buf[0:self.offset]
        return b''.join(self.buf)


class CollectorProtocol(Protocol):
    def __init__(self, factory, msg_handler=None):
        self.factory = factory
        self.state = const.NET_STATUS.WAIT
        self.remoteid = None
        self.nodeid = self.factory.nodeid
        self.msg_handler = msg_handler
        self.buffer = FastBufferString()
        self.current_max_iter = 0
        self.is_running = True
        self.msg_queue = Queue.Queue()

    def msg_proceess_thread(self):
        while True:
            msg = self.msg_queue.get()
            if not self.is_running:
                break
            self.processData(msg)

    def connectionMade(self):
        peer = self.transport.getPeer()
        #self.factory.peers[peer] = peer
        self.factory.counter += 1
        if self.factory.is_multithreading:
            self.mt = Thread(name='processthread', target=self.msg_proceess_thread)
            self.mt.start()
        logger.info("Connection from : %s, online clients: %d", peer, self.factory.counter)

    def connectionLost(self, reason):
        if self.remoteid in self.factory.peers:
            self.factory.peers.pop(self.remoteid, None)
        #self.factory.lock.acquire()
        if self.remoteid in self.factory.accs:
            self.factory.accs.pop(self.remoteid, None)
        #self.factory.lock.release()
        if self.msg_handler:
            self.msg_handler.handle_connection_lost(self, reason)
        logger.info('%s disconnected', self.transport.getPeer())
        self.factory.counter -= 1
        if self.factory.is_multithreading:
            self.is_running = False
            self.msg_queue.put(None)
        logger.info('Online clients: %d', self.factory.counter)

    def genExchangePolicy(self, accs):
        active_set = []
        passive_set = []
        exchange = {}
        max_iter = 0
        if len(accs) == 0:
            return exchange, max_iter
        accs_copy = dict(accs)
        for dk in accs_copy:
            d = accs_copy[dk]
            rrank = d['rank']
            train_host = d['train_host']
            role = d['role']
            iter = d['iter']
            if d['sent'] > 1000:
                logger.info('The client %s seems timeout' % train_host)
                accs.pop(dk, None)
                continue
            if role == ROLE.ACTIVE:
                active_set.append(train_host)
            else:
                passive_set.append(train_host)
            if max_iter < iter:
                max_iter = iter
            accs[dk]['sent'] += 1
        if len(passive_set) == 0:
            return exchange, max_iter
        x = [i for i in range(len(passive_set))]
        shuffle(x)
        for i, th in enumerate(active_set):
            exchange[th] = passive_set[x[i%len(x)]]
        return exchange, max_iter

    def processData(self, data):
        #logger.info("recv time: ", str(datetime.now()))
        #logger.info("Msg recved len:", len(data), data[-10:])
        for line in data.split(b'EEEE'):
            #logger.info("line string:", line[-4:])
            #logger.info("split: ", str(datetime.now()))
            #line = line.strip()
            is_model = False
            if line[-4:] != b'::::' and line[-4:] != b'++++':
                self.buffer.insert(line)
                #logger.info("check: ", str(datetime.now()))
                return
            else:
                if line[-4:] == b'++++':
                    is_model = True
                self.buffer.insert(line[:-4])
            buffer = self.buffer.get() #+ line
            s = time.time()
            if is_model:
                #logger.info('Recved model')
                msg = self.decode_model_msg(buffer)
                #logger.info('msgtype: ', msg['msgtype'])
            else:
                #msg = json.loads(buffer)
                msg = pickle.loads(buffer)
            #logger.info("Parse time: ", time.time()-s)
            self.buffer.clear()
            msgtype = msg['msgtype']
            remoteid = msg['nodeid']
            #print 'msg: ', msg
            if msgtype == const.CL_MSG_TYPE.HI: 
                self.remoteid = remoteid 
                self.factory.peers[self.remoteid] = self
                if self.msg_handler:
                    self.msg_handler.handle_hi(self, msg)
            elif msgtype == const.CL_MSG_TYPE.UPDATE_ACC:
                self.factory.lock.acquire()
                msg['data']['sent'] = 0
                self.factory.accs[remoteid] = msg['data'] 
                accs = self.factory.accs
                self.factory.lock.release()

                #print 'updated accs: ', self.factory.accs
                #for rid in self.factory.peers:
                #    p = self.factory.peers[rid]
                resp_msg = {}
                resp_msg['data'] = accs
                exchange, max_iter = self.genExchangePolicy(accs)
                resp_msg['exchange'] = exchange 
                resp_msg['msgtype'] = const.CL_MSG_TYPE.GET_ACCS_REPS
                self.send_msg(resp_msg)

                #if max_iter != self.current_max_iter and max_iter == 1562 * 10: #or max_iter == 1562 * 30:
                #    self.current_max_iter = max_iter
                #    for rid in self.factory.peers:
                #        p = self.factory.peers[rid]
                #        new_msg = {}
                #        new_msg['msgtype'] = const.CL_MSG_TYPE.CHANGE_ROLE
                #        p.send_msg(new_msg)

            elif msgtype == const.CL_MSG_TYPE.GET_ACCS:
                p = self.factory.peers[remoteid]
                resp_msg = {}
                #self.factory.lock.acquire()
                resp_msg['data'] = dict(self.factory.accs)
                #self.factory.lock.release()
                resp_msg['msgtype'] = const.CL_MSG_TYPE.GET_ACCS_REPS
                p.send_msg(resp_msg)
            elif msgtype == const.CL_MSG_TYPE.GET_ACCS_REPS:
                #print 'recved reps: ', msg
                if self.msg_handler:
                    self.msg_handler.get_accs_reps(self, msg)
            elif msgtype == const.CL_MSG_TYPE.CHANGE_ROLE:
                if self.msg_handler:
                    self.msg_handler.change_role_reps(self, msg)
            elif msgtype == const.PEER_MSG_TYPE.MODEL_REQ:
                #logger.info('[PEER MSG] ask for model: ', msg)
                #p = self.factory.peers[remoteid]
                #resp_msg = {}
                #resp_msg['train_host'] = self.factory.get_train_host()
                #resp_msg['msgtype'] = const.PEER_MSG_TYPE.MODEL_REPS
                #p.send_msg(resp_msg)
                if self.msg_handler:
                    self.msg_handler.handle_modelreq(self, msg)
            elif msgtype == const.PEER_MSG_TYPE.MODEL_REPS:
                #logger.info( '[PEER MSG] model response: ', msg)
                if self.msg_handler:
                    self.msg_handler.get_model_reps(self, msg)
            #print 'peers: ', self.factory.peers

    def dataReceived(self, data):
        if self.factory.is_multithreading:
            logger.info("before put: %s", str(datetime.now()))
            self.msg_queue.put(data)
            logger.info("after put: %s", str(datetime.now()))
        else:
            self.processData(data)

    def send_msg(self, msg):
        msg['nodeid'] = self.nodeid
        d = b''.join([pickle.dumps(msg), b"::::EEEE"])
        #d = b''.join([json.dumps(msg).encode('utf-8'), b"::::EEEE"])
        self.transport.write(d)

    def send_model(self, model, loss, is_asked=1):
        msg = {}
        msg['train_host'] = self.factory.get_train_host()
        msg['model'] = model
        msg['loss'] = loss
        msg['msgtype'] = const.PEER_MSG_TYPE.MODEL_REPS
        msg['is_asked'] = is_asked 
        msg['nodeid'] = self.nodeid
        encode_msg = self.encode_model_msg(msg)
        #logger.info('len of model msg: ', len(encode_msg))
        self.transport.write(encode_msg)

    def decode_model_msg(self, msg):
        #t = time.time()
        ret = {}
        length = struct.unpack('i',msg[0:4])[0]
        offset = 4
        ret['train_host'] = msg[offset:offset+length].decode('utf-8')
        offset += length
        length = struct.unpack('i',msg[offset:offset+4])[0]
        offset += 4
        ret['model'] = msg[offset:offset+length]
        offset += length
        ret['loss'] = struct.unpack('f',msg[offset:offset+4])[0]
        offset += 4
        length = struct.unpack('i',msg[offset:offset+4])[0]
        offset += 4
        ret['msgtype'] = msg[offset:offset+length].decode('utf-8')
        offset += length
        ret['is_asked'] = struct.unpack('i', msg[offset:offset+4])[0]
        offset += 4
        length = struct.unpack('i',msg[offset:offset+4])[0]
        offset += 4
        ret['nodeid'] = msg[offset:offset+length].decode('utf-8')
        #logger.info('decode_model_msg time used: ', time.time()-t)
        return ret

    def encode_model_msg(self, msg):
        """
        train_host:int[length of train_host]train_host
        model:int[length of model]model
        loss:float[loss in float]
        msgtype:int[length of msgtype]msgtype
        is_asked:int
        nodeid:int[length of nodeid]nodeid
        """
        t = time.time()
        l = [struct.pack('i',len(msg['train_host'].encode('utf-8'))),
            msg['train_host'].encode('utf-8'),
            struct.pack('i',len(msg['model'])),
            msg['model'],
            struct.pack('f', msg['loss']),
            struct.pack('i', len(msg['msgtype'].encode('utf-8'))),
            msg['msgtype'].encode('utf-8'),
            struct.pack('i', msg['is_asked']),
            struct.pack('i', len(msg['nodeid'].encode('utf-8'))),
            msg['nodeid'].encode('utf-8'),
            ]
        l.append(b'++++EEEE')
        s = b''.join(l)
        #logger.info('encode_model_msg time used: ', time.time()-t)
        return s

    def send_accuracy(self, accuracy, rank, role, iter):
        msg = {}
        data = {}
        data['accuracy'] = accuracy
        data['rank'] = rank 
        data['train_host'] = self.factory.get_train_host()
        data['role'] = role
        data['iter'] = iter
        msg['data'] = data
        msg['msgtype'] = const.CL_MSG_TYPE.UPDATE_ACC
        # test numpy array
        #n = np.array([[20.0, 30.0], [2.0, 4.0]], dtype=np.float32)
        #msg['model'] = n.tolist()
        self.send_msg(msg)

    def send_get_accs_req(self):
        msg = {}
        msg['msgtype'] = const.CL_MSG_TYPE.GET_ACCS
        self.send_msg(msg)

    def send_get_model_req(self):
        msg = {}
        msg['msgtype'] = const.PEER_MSG_TYPE.MODEL_REQ
        self.send_msg(msg)


class PSCollectorProtocol(CollectorProtocol, object):
    def __init__(self, factory, msg_handler=None):
        super(PSCollectorProtocol, self).__init__(factory, msg_handler)

    def processData(self, data):
        for line in data.split(b'EEEE'):
            is_model = False
            if line[-4:] != b'::::' and line[-4:] != b'++++':
                self.buffer.insert(line)
                return
            else:
                if line[-4:] == b'++++':
                    is_model = True
                self.buffer.insert(line[:-4])
            buffer = self.buffer.get() #+ line
            s = time.time()
            if is_model:
                msg = self.decode_model_msg(buffer)
            else:
                msg = pickle.loads(buffer)
            self.buffer.clear()
            msgtype = msg['msgtype']
            remoteid = msg['nodeid']
            if msgtype == const.CL_MSG_TYPE.HI: 
                self.remoteid = remoteid 
                self.factory.peers[self.remoteid] = self
                if self.msg_handler:
                    self.msg_handler.handle_hi(self, msg)
            elif msgtype == const.PEER_MSG_TYPE.MODEL_REQ:
                if self.msg_handler:
                    self.msg_handler.handle_modelreq(self, msg)
            elif msgtype == const.PEER_MSG_TYPE.MODEL_REPS:
                if self.msg_handler:
                    self.msg_handler.get_model_reps(self, msg)



class CollectorFactory(Factory):
    def __init__(self, train_ip, train_port, msg_handler=None, is_multithreading=False):
        self.msg_handler = msg_handler 
        self.train_ip = train_ip
        self.train_port = train_port
        self.protocols = {} 
        self.nodeid = generate_nodeid()
        self.peers = {}
        self.counter = 0
        self.accs = {}
        self.is_multithreading = is_multithreading
        self.lock = Lock()

    def get_train_host(self):
        return self.train_ip + ':' + str(self.train_port)

    def startFactory(self):
        logger.info('startFactory: %s', self.nodeid)

    def buildProtocol(self, addr):
        logger.info('addr: %s', addr)
        self.protocol = CollectorProtocol(self, self.msg_handler)
        self.protocols[addr] = self.protocol
        return self.protocol

    def clientConnectionLost(self, connector, reason):
        connector.connect()

    def clientConnectionFailed(self, connector, reason):
        logger.info("connection failed: %s", reason)
        connector.connect()
        #reactor.stop()

class PSCollectorFactory(CollectorFactory, object):
    def __init__(self, train_ip, train_port, dnn, lr, dataset, msg_handler=None, is_multithreading=False):
        super(PSCollectorFactory, self).__init__(train_ip, train_port, msg_handler=self, is_multithreading=is_multithreading)
        self.from_others_connections = {}
        self.initialize_dnn(dnn, lr, dataset)

    def initialize_dnn(self, dnn, lr, dataset):
        self.trainer = DLTrainer(0, 1, dist=False, batch_size=64, is_weak_scaling=True, ngpus=0, data_dir=None, dataset=dataset, dnn=dnn, lr=lr, nworkers=1, prefix='parameterserver')

    def buildProtocol(self, addr):
        logger.info('addr: %s', addr)
        self.protocol = PSCollectorProtocol(self, self.msg_handler)
        self.protocols[addr] = self.protocol
        return self.protocol

    def handle_hi(self, p, msg):
        self.from_others_connections[msg['train_host']] = p

    def handle_connection_lost(self, p, reason):
        host = None
        for h in self.from_others_connections:
            if p == self.from_others_connections.get(h, None):
                host = h
                break
        self.from_others_connections.pop(host, None)

    def handle_modelreq(self, p, msg):
        pass
        #model = self.trainer.get_model('MODEL')
        #loss = 0.
        #recved_gradients = msg['model']
        #logger.info('recieved gradients.')
        #self.trainer.update_with_remote_gradients(recved_gradients)
        #p.send_model(model, loss, is_asked=1)

    def get_model_reps(self, p, msg):
        #is_asked = msg['is_asked']
        #logger.info('Recieved gradients from %s', p.transport.getPeer())
        recved_gradients = msg['model']
        #recved_loss = msg['loss']
        # This should be the gradients
        self.trainer.update_with_remote_gradients(recved_gradients)
        model = self.trainer.get_model('MODEL')
        loss = 0.
        #logger.info('Send model to client: %s', p.transport.getPeer())
        p.send_model(model, loss, is_asked=1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="trainer collector")
    parser.add_argument('--port', type=int, default=settings.SERVER_PORT)
    parser.add_argument('--log', type=str, default='./logs/server.log')
    parser.add_argument('--listen', default=settings.SERVER_IP)
    subparsers = parser.add_subparsers(help='[ps]', dest='func')
    coo = subparsers.add_parser('coo', help='Run as a coordinator')
    ps = subparsers.add_parser('ps', help='Run as a parameter server')
    ps.add_argument('--dnn', type=str, default='resnet50', choices=['resnet50', 'resnet20', 'vgg19', 'vgg16', 'mnistnet', 'alexnet'], help='Specify the neural network for training')
    ps.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar10'], help='Specify the dataset for training')
    ps.add_argument('--lr', type=float, default=0.1, help='Default learning rate')
    args = parser.parse_args()
    bind = args.listen
    port = args.port
    logfile = args.log
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', args)

    if args.func == 'coo':
        factory = CollectorFactory(bind, port, is_multithreading=False)
    else:
        factory = PSCollectorFactory(bind, port, args.dnn, args.lr, args.dataset, is_multithreading=False)
    endpoint = TCP4ServerEndpoint(reactor, args.port)
    endpoint.listen(factory)

    reactor.run()

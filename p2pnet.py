# -*- coding: utf-8 -*-
from __future__ import print_function
from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.internet import epollreactor
epollreactor.install()
from twisted.internet import reactor, defer
from csp2p import constants
from csp2p.collector import CollectorProtocol, CollectorFactory
import socket
import settings
import simplejson as json
import time
import threading

from settings import logger

collector_server_addr = {
            'addr': settings.SERVER_IP,
            'port': settings.SERVER_PORT}

TIMEOUT=60


class P2PNet():
    def __init__(self, rank, bind, port=settings.PORT, callback=None, is_psworker=False):
        #hostname = socket.gethostname()
        ip = bind #hostname
        self.ip_ = ip
        self.port_ = port
        self.rank_ = rank
        self.collector_server_addr = collector_server_addr
        self.collector_server_protocal = None
        self.train_servers = {}
        self.from_others_connections = {}

        self.rank_to_remoteid = {}
        self.remoteid_to_rank = {}
        self.acc_dict = {}
        self.timer_tasks = {}
        self.is_psworker = is_psworker
        self.callback = callback

        self.lock = threading.Lock()
        self.d = defer.Deferred()
        logger.info(str(self.ip_)+':'+str(self.port_))

    def run(self):
        # Start trainer server, to recieve the model request
        trainer_factory = CollectorFactory(self.ip_, self.port_, self, is_multithreading=False)
        endpoint = TCP4ServerEndpoint(reactor, self.port_)
        endpoint.listen(trainer_factory)
        
        client_factory = CollectorFactory(self.ip_, self.port_, self)
        point = TCP4ClientEndpoint(reactor, self.collector_server_addr['addr'], self.collector_server_addr['port'])
        client_protocol = CollectorProtocol(client_factory, self)
        d = connectProtocol(point, client_protocol)
        d.addCallback(self.connect_to_collector_protocol)

        #reactor.callLater(5, self.send_accuracy, 0.9, self.rank_)
        #reactor.callLater(10, self.send_accuracy, 0.95, self.rank_)
        #reactor.callLater(20, self.send_get_accs_req, )
        #reactor.callLater(30, self.send_get_accs_req, )

        reactor.run(installSignalHandlers=False)
        #reactor.run()

    def handle_hi(self, p, msg):
        self.from_others_connections[msg['train_host']] = p
        logger.info('self.from_others_connections: %s', self.from_others_connections)

    def handle_connection_lost(self, p, reason):
        host = None
        for h in self.train_servers:
            if p == self.train_servers.get(h, None):
                host = h
                break
        logger.debug('train servers: %s', self.train_servers)
        logger.info('handle_connection_lost protocol: %s', p)
        logger.debug('host: %s', host)
        if host:
            logger.info('train server connection lost: %s', host)
            self.train_servers.pop(host)
            task_id = self.timer_tasks.get(host, None)
            if task_id:
                if task_id.active():
                    task_id.cancel()
                self.timer_tasks.pop(host, None)
            if self.callback:
                self.callback.train_server_disconnected(p)
        host = None
        for h in self.from_others_connections:
            if p == self.from_others_connections.get(h, None):
                host = h
                break
        if host:
            self.from_others_connections.pop(host)
            logger.info('client connection lost: %s', host)

    def handle_modelreq(self, p, msg):
        #logger.debug('Recieve model request from %s', msg)
        logger.debug('Recieve model request from %s', msg['train_host'])
        if self.callback:
            model, loss = self.callback.get_local_model()
            s = time.time()
            p.send_model(model, loss, is_asked=1)
            logger.debug('Dumps model msg time used %f', time.time()-s)
            self.callback.handle_modelreq(p, msg)
            
    def disconnected(self, remoteid):
        #del self.boardcasts[self.remoteid_to_rank[remoteid]]
        #del self.protocols[remoteid]
        pass

    def connect_to_collector_protocol(self, p):
        """The callback to start the protocol exchange. We let connecting
        nodes start the hello handshake""" 
        self.collector_server_protocal = p 
        if self.is_psworker:
            host = self.collector_server_addr['addr']+':'+str(self.collector_server_addr['port'])
            self.train_servers[host] = p 
        msg = {'rank': self.rank_, 'msgtype': constants.CL_MSG_TYPE.HI, 'train_host': self.get_local_host()}
        p.send_msg(msg)

    def connect_to_trainer(self, remote_rank, train_host, model_loss=None):
        host, port = train_host.split(':')
        logger.info('train_host is not in the train-servers %s, %s', train_host, self.train_servers)
        logger.debug('host:%s, port: %s', host, port)

        client_factory = CollectorFactory(self.ip_, self.port_, self, is_multithreading=False)
        point = TCP4ClientEndpoint(reactor, host, int(port))
        client_protocol = CollectorProtocol(client_factory, self)
        d = connectProtocol(point, client_protocol)
        d.addCallback(self.connect_to_trainer_protocol, train_host, remote_rank, model_loss)
        d.addErrback(self.connect_to_trainer_protocol_error, train_host, remote_rank)
        task_id = reactor.callLater(TIMEOUT, self.ask_model_timeout, train_host)
        self.timer_tasks[train_host] = task_id
        logger.info('Trying to connect server %s', train_host)

    def connect_to_trainer_protocol(self, p, host, remote_rank, model_loss):
        logger.debug('connect to server %s successfully', host)
        task_id = self.timer_tasks.get(host, None)
        if task_id:
            if task_id.active():
                task_id.cancel()
            self.timer_tasks.pop(task_id, None)
        self.train_servers[host] = p 
        msg = {'rank': self.rank_, 'msgtype': constants.CL_MSG_TYPE.HI, 'train_host': self.get_local_host()}
        p.send_msg(msg)
        if model_loss:
            self.ask_model_from(remote_rank, host, model_loss)
        logger.info('self.train_servers: %s', self.train_servers)

    def connect_to_trainer_protocol_error(self, p, host, remote_rank):
        logger.info('connect to server %s failed!', host)
        if self.callback:
            self.callback.train_server_disconnected(p)

    def send_accuracy(self, acc, rank, role, iter):
        if self.collector_server_protocal:
            #logger.info('send acc')
            self.collector_server_protocal.send_accuracy(acc, rank, role, iter)

    # -> get_accs
    def send_get_accs_req(self):
        self.collector_server_protocal.send_get_accs_req()

    def get_accs_reps(self, conn_protocal, msg):
        if conn_protocal == self.collector_server_protocal:
            #logger.info('get acc resp')
            self.acc_dict = msg

            #data = msg.get('data', [])
            #for dk in data:
            #    d = data[dk]
            #    rrank = d['rank']
            #    train_host = d['train_host']
            #    role = d['role']
            #    if train_host not in self.train_servers and train_host != self.get_local_host() and role == constants.ROLE.PASSIVE:
            #        self.connect_to_trainer(rrank, train_host, None)

            if self.callback:
                self.callback.get_accs_reps(conn_protocal, msg)

    # <- get_accs End
    def change_role_reps(self, conn_protocal, msg):
        logger.info('change role msg recieved')
        if self.callback:
            self.callback.handle_change_role(msg)

    def get_accs(self):
        return dict(self.acc_dict)

    # -> get_model
    def send_get_model_req(self, p, model_loss):
        msg = {}
        msg['msgtype'] = constants.PEER_MSG_TYPE.MODEL_REQ
        msg['rank'] = self.rank_
        msg['host'] = self.get_local_host()
        if self.callback:
            #if model_loss:
            #    model = model_loss[0]
            #    loss = model_loss[1]
            #else:
            #    model, loss = self.callback.get_local_model()
            #msg['model'] = model
            #msg['loss'] = loss
            #msg['is_asked'] = 0
            msg['train_host'] = self.get_local_host()
        p.send_msg(msg)

    def get_model_reps(self, conn_protocal, msg):
        #logger.info('get_model_reps')
        task_id = self.timer_tasks.get(msg['train_host'], None)
        if task_id:
            if task_id.active():
                task_id.cancel()
            self.timer_tasks.pop(task_id, None)
        if self.callback:
            self.callback.model_transfering(msg)
        if not msg['is_asked']:
            if self.callback:
                model, loss = self.callback.get_local_model()
                s = time.time()
                conn_protocal.send_model(model, loss, is_asked=1)
                logger.debug('Dumps model msg time used %f', time.time()-s)
    # <- get_model End

    def get_local_host(self):
        host = self.ip_ + ':' + str(self.port_)
        return host

    def ask_model_timeout(self, train_host):
        logger.info('Ask model from server %s timeout!', train_host)
        if self.callback:
            self.callback.train_server_disconnected(None)
        self.train_servers.pop(train_host, None)
        self.timer_tasks.pop(train_host, None)

    def ask_model_from(self, remote_rank, train_host, model_loss=None):
        if train_host not in self.train_servers:
            #self.callback.model_transfer_lock.release()
            self.connect_to_trainer(remote_rank, train_host, model_loss)
        else:
            if settings.ACTIVE_WAIT and self.callback.model_transfer_lock:
                self.callback.model_transfer_lock.acquire()
            logger.debug('ask model from: %s', train_host)
            p = self.train_servers[train_host]
            s = time.time()
            #self.send_get_model_req(p, model_loss)
            #if model_loss:
            #logger.debug('Procotol: %s', p)
            p.send_model(model_loss[0], model_loss[1], is_asked=0)
            logger.debug('Dumps model msg time used %f', time.time()-s)

            task_id = reactor.callLater(TIMEOUT, self.ask_model_timeout, train_host)
            self.timer_tasks[train_host] = task_id

            #if self.callback: # send model to the node that needs to require the model
            #    model, loss = self.callback.get_local_model()
            #    p.send_model(model, loss, is_asked=0)

    def stop(self):
        reactor.stop()



if __name__ == '__main__':
    hostname = socket.gethostname()
    rank = 0
    try:
        rank = settings.BOOTSTRAP_LIST.index(hostname)
    except:
        logger.info('Please add the host into the BOOTSTRAP_LIST in settings.py')
        exit(1)
    p2pnet = P2PNet(rank)
    p2pnet.run()


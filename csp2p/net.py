from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.internet.protocol import Protocol, Factory
from twisted.internet import reactor
from uuid import uuid4
import constants as const
import json
import time
import argparse

generate_nodeid = lambda: str(uuid4())

class MyProtocol(Protocol):
    def __init__(self, factory, msg_handler=None):
        self.factory = factory
        self.state = const.NET_STATUS.WAIT
        self.remote_nodeid = None
        self.nodeid = self.factory.nodeid
        self.msg_handler = msg_handler

    def connectionMade(self):
        peer = self.transport.getPeer()
        #self.factory.peers[peer] = peer
        print "Connection from : ", peer

    def connectionLost(self, reason):
        #if self.remote_nodeid in self.factory.peers:
        #    self.factory.peers.pop(self.remote_nodeid)
        #print('lose connecting: ', self.remote_nodeid, self.nodeid)
        #if self.remote_nodeid != self.nodeid:
        #    if self.msg_handler:
        #        self.msg_handler.disconnected(self.remote_nodeid)
        print self.nodeid, "disconnected"
        print self, "disconnected"

    def dataReceived(self, data):
        for line in data.splitlines():
            line = line.strip()
            msg = json.loads(line)
            self.remote_nodeid = msg['nodeid']
            self.factory.peers[self.remote_nodeid] = self
            print 'peers: ', self.factory.peers
            self.state = const.NET_STATUS.READY
            if self.msg_handler:
                self.msg_handler.handle_msg(self, msg)

    def send_msg(self, msg):
        msg['nodeid'] = self.nodeid
        print 'send_msg: ', msg
        self.transport.write(json.dumps(msg)+ "\n")


class MyFactory(Factory):
    def __init__(self, msg_handler=None):
        self.msg_handler = msg_handler 
        self.protocols = {} 
        self.nodeid = generate_nodeid()
        self.peers = {}

    def startFactory(self):
        print('startFactory: ', self.nodeid)

    def buildProtocol(self, addr):
        print('addr: ', str(addr))
        print('addr: ', addr.exploded)
        self.protocol = MyProtocol(self, self.msg_handler)
        self.protocols[addr] = self.protocol
        return self.protocol

def gotProtocol(p):
    """The callback to start the protocol exchange. We let connecting
    nodes start the hello handshake""" 
    print('Protocol got')
    msg = {'msgtype': const.MSG_TYPE.HI}
    p.send_msg(msg)


DEFAULT_PORT = 5991
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ncpoc")
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--listen', default="127.0.0.1")
    parser.add_argument('--bootstrap', action="append", default=['gpu20', 'gpu21'])
    args = parser.parse_args()
    try:
        factory = MyFactory()
        endpoint = TCP4ServerEndpoint(reactor, args.port)
        endpoint.listen(factory)
    except Exception as e:
        print("[!] Address in use", e)
        raise SystemExit

    #BOOTSTRAP_LIST = [str(ip)+':'+str(args.port) for ip in args.bootstrap]
    #protocols = []
    #for bootstrap in BOOTSTRAP_LIST:
    #    host, port = bootstrap.split(":")
    #    point = TCP4ClientEndpoint(reactor, host, int(port))
    #    p = MyProtocol(factory)
    #    d = connectProtocol(point, p)
    #    d.addCallback(gotProtocol)
    #    protocols.append(p)
    reactor.run()

from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
from twisted.internet.protocol import ClientFactory
from twisted.internet import reactor
from net import MyProtocol, MyFactory, gotProtocol, DEFAULT_PORT, generate_nodeid


protocols = []
factory = MyFactory()
host = 'gpu20'
port = DEFAULT_PORT
point = TCP4ClientEndpoint(reactor, host, int(port))
p = MyProtocol(factory)
d = connectProtocol(point, p)
d.addCallback(gotProtocol)

reactor.run()



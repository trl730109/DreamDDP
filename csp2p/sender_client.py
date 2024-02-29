from twisted.internet.endpoints import TCP4ClientEndpoint, connectProtocol
import socket
from twisted.internet.protocol import ClientFactory
from twisted.internet import reactor
from collector import CollectorProtocol, CollectorFactory, gotProtocol, DEFAULT_PORT, generate_nodeid


protocols = []
host = 'gpu20'
myhost = socket.gethostname()
port = DEFAULT_PORT
factory = CollectorFactory(myhost, port)
point = TCP4ClientEndpoint(reactor, host, int(port))
p = CollectorProtocol(factory)
d = connectProtocol(point, p)
d.addCallback(gotProtocol)

reactor.callLater(5, p.send_accuracy, 0.9)
reactor.callLater(10, p.send_accuracy, 0.95)
reactor.callLater(20, p.send_get_accs_req, )

reactor.run()



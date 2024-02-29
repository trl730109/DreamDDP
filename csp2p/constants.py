
class NET_STATUS:
    WAIT = 'WAIT'
    READY = 'READY'

class MSG_TYPE:
    HI = 'hello'
    PING = 'ping'
    PONG = 'pong'
    BCST = 'broadcast'

class CL_MSG_TYPE:
    HI = 'hi'
    UPDATE_ACC = 'udpate_acc'
    GET_ACCS = 'get_accs'
    GET_ACCS_REPS = 'get_accs_reps'
    CHANGE_ROLE = 'change_role'

class PEER_MSG_TYPE:
    MODEL_REQ = 'mdoelreq'
    MODEL_REPS = 'mdoelreps'

class ROLE:
    ACTIVE = 'active'
    PASSIVE = 'passive'

class PROCESS_MSG_TYPE:
    SEND_ACC = 'sendacc'
    RECV_ACC = 'recvacc'
    SEND_MODELREQ = 'sendmodelreq'
    GET_MODEL = 'getmodel'
    RECV_MODEL = 'recvmodel'
    EXIT = 'exit'

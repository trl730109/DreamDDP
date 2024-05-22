import os
import logging
#import coloredlogs
#coloredlogs.install()
import socket

DEBUG =0
#SERVER_IP = '127.0.0.1'
SERVER_IP = '192.168.0.20'
#SERVER_IP = '158.182.9.51' # 1 node 8 workers
#SERVER_IP = '158.182.9.51' # 2 nodes 16 workers
#SERVER_IP = '158.182.9.53'  # 4 nodes 32 workers
#SERVER_IP = '158.182.9.50'  # 8 nodes 64 workers
#SERVER_IP = '158.182.9.78'  # 8 nodes 64 workers
#SERVER_IP = '158.182.9.40' # 16 node 128 workers
#SERVER_IP = '158.182.9.50'  # 32 nodes 256 workers
SERVER_PORT = 5911
PORT = 5922
ACTIVE_WAIT=False
PASSIVE_WAIT=False
GPU_CONSTRUCTION=True
WARMUP=True
ZHU=False
PS=False
if PS:
    ACTIVE_WAIT=True
DELAY_COMM=1

PREFIX=''
if WARMUP:
    PREFIX=PREFIX+'gwarmup'
if ACTIVE_WAIT:
    PREFIX=PREFIX+'-wait'
if PS:
    PREFIX=PREFIX+'-ps'


PREFIX=PREFIX+'-dc'+str(DELAY_COMM)
#EXCHANGE_MODE = 'MODEL' 
#EXCHANGE_MODE = 'MODEL+GRAD' 
EXCHANGE_MODE = 'MODEL' 

LOGGING_ASSUMPTION=False
LOGGING_GRADIENTS=False
PREFIX=PREFIX+'-'+EXCHANGE_MODE.lower()

#EXP='-iclr-accuracy'
#EXP='-iclr-gradients'
#EXP='-posticlr-accuracy'
#EXP='-posticlr-speed'
#EXP='-iclr-debug'
#EXP='-tpds10GbE-v2-r1'
#EXP='-tpds56GbIB-v2-r3'
#EXP='-tpds-convergence'
#EXP='-debug'
#EXP='-exp'
#EXP='-nips2020'
EXP='-debug'
#EXP='-nips2020-r5'
#EXP='-imagenet-convergence'
FP16=False
ADAPTIVE_MERGE=False
ADAPTIVE_SPARSE=False
if FP16:
    EXP=EXP+'-fp16'
PREFIX=PREFIX+EXP
if ADAPTIVE_MERGE:
    PREFIX=PREFIX+'-ada'

FAKE_DATA=False
ORIGINAL_HOROVOD=False
EXCHANGE_PARA=False
if ORIGINAL_HOROVOD:
    PREFIX=PREFIX+'-hvd'

#EFFICIENT_IO=True
EFFICIENT_IO=False

TENSORBOARD=False
if ZHU:
    PREFIX=PREFIX+'-zhu'
DELAY=0

MAX_EPOCHS = 200
#MAX_EPOCHS = 90

BOOTSTRAP_LIST = ['gpu10', 'gpu11']#, 'gpu12', 'gpu13']

hostname = socket.gethostname() 
logger = logging.getLogger(hostname)

if DEBUG:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

strhdlr = logging.StreamHandler()
logger.addHandler(strhdlr)
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s')
strhdlr.setFormatter(formatter)
#PYTHONPATH=./ python csp2p/collector.py --log logs/server.log coo
PYTHONPATH=./ python csp2p/collector.py --log logs/server.log ps --dnn resnet20 --lr 0.1 --dataset cifar10 

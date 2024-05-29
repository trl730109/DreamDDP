#ssh mgd@10.0.1.15 "cd /home/mgd/p/p2p-dl; ./syncfrommgd3.sh"
rsync -u -avz -r mgd@10.0.1.18:/home/mgd/p/p2p-dl/logs/* ./logs/

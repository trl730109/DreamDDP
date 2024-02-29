#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-1}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"
nwpernode=4
nstepsupdate=16
rdma="${rdma:-0}"
MPIPATH=/home/esetstore/.local/openmpi-4.0.1
PY=/home/esetstore/pytorch1.4/bin/python
#PY=/home/esetstore/shshi/anaconda3/bin/python
#PY=/localdata/anaconda3/bin/python
#PY=/home/comp/csshshi/anaconda2/bin/python
#PY=/usr/local/bin/python
#PY=/home/comp/csshshi/anaconda2/bin/python
#MPIPATH=/usr/local/openmpi/openmpi-4.0.1
#MPIPATH=/home/comp/csshshi/local/openmpi3.1.1
#MPIPATH=/home/esetstore/.local/openmpi-4.0.1
#PY=/home/esetstore/anaconda3/bin/python
#PY=python3
GRADSPATH=./logs/gradients


if [ "$rdma" = "0" ]; then
params="-mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include 192.168.0.1/24 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_CACHE_CAPACITY=0"
else
params="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    -mca btl_tcp_if_include ib0 \
    --mca btl_openib_want_fork_support 1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_SOCKET_IFNAME=ib0 \
    -x NCCL_DEBUG=INFO"
fi


$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cj-cluster${nworkers} -bind-to none -map-by slot \
    $params \
    $PY horovod_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --threshold $threshold --saved-dir $GRADSPATH --momentum-correction $momentum_correction
#    --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
#    -mca btl_tcp_if_include ib0 \
#    --mca btl_openib_want_fork_support 1 \
#    -x LD_LIBRARY_PATH  \
#    -x NCCL_IB_DISABLE=0 \
#    -x NCCL_DEBUG=INFO \
#    -x NCCL_SOCKET_IFNAME=ib0 \
#    -x NCCL_TREE_THRESHOLD=0 \


#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-1.0}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"
nwpernode=4
nstepsupdate=1
#PY=/home/comp/csshshi/anaconda2/bin/python
PY=/usr/local/bin/python
#PY=/home/comp/csshshi/anaconda2/bin/python
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
#MPIPATH=/home/comp/csshshi/local/openmpi3.1.1
#MPIPATH=/home/esetstore/.local/openmpi-4.0.1
#PY=/home/esetstore/anaconda3/bin/python
#PY=python3
#PY=/home/comp/15485625/pytorch1.4/bin/python
GRADSPATH=./logs/iclr

#cd $CURR_PATH
#HOROVOD_CYCLE_TIME=1 HOROVOD_FUSION_THRESHOLD=0 
#HOROVOD_TIMELINE=./logs/profile-timeline-${dnn}.json.log HOROVOD_TIMELINE_MARK_CYCLES=1 

$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include em1 \
    -x LD_LIBRARY_PATH  \
    -x NCCL_IB_DISABLE=1 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=em1 \
    -x NCCL_TREE_THRESHOLD=0 \
    $PY horovod_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --threshold $threshold --saved-dir $GRADSPATH --momentum-correction $momentum_correction
    #-mca pml ob1 -mca btl openib -mca btl_openib_allow_ib 1 \
    #-x NCCL_IB_DISABLE=0 \
    #-x NCCL_DEBUG=INFO \
    #--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
    #-mca btl_tcp_if_include eth0 \
    #-x NCCL_SOCKET_IFNAME=eth0 \
    #-mca pml ob1 -mca btl ^openib \
    #-x NCCL_IB_DISABLE=1 \
    #-mca btl_tcp_if_include em1 \
    #-x HOROVOD_FUSION_THRESHOLD=0 \
    #-x NCCL_SOCKET_IFNAME=em1 \
    #-x NCCL_DEBUG=INFO \
    #-mca btl_tcp_if_include bond0 \
    #-mca btl_tcp_if_include eth0 \
    #-x NCCL_SOCKET_IFNAME=eth0 \
    #-mca btl_tcp_if_include eth0 \
    #-x NCCL_SHM_DISABLE=1 \
    #-x NCCL_P2P_DISABLE=1 \
    #-x NCCL_CHECKS_DISABLE=1 \

#!/bin/bash
CURR_PATH=/home/shaohuais/repos/p2p-topk
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-0.001}"
threshold="${threshold:-0}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"
nwpernode=4
nstepsupdate=1
#PY=/home/comp/csshshi/anaconda2/bin/python
#MPIPATH=/home/comp/csshshi/local/openmpi3.1.1
MPIPATH=/home/shaohuais/.local/openmpi-4.0.0
PY=/usr/bin/python3.5
#GRADSPATH=/datasets/shshi/iclr
GRADSPATH=./logs/iclr

#HOROVOD_TIMELINE=./logs/profile-timeline.json.log HOROVOD_TIMELINE_MARK_CYCLES=1 
#cd $CURR_PATH
#HOROVOD_CYCLE_TIME=1 HOROVOD_FUSION_THRESHOLD=0 
    #-x NCCL_LL_THRESHOLD=16 \
    #-x HOROVOD_MPI_THREADS_DISABLE=1 \
    #-x HOROVOD_CYCLE_TIME=1  \
    #-x NCCL_BUFFSIZE=16777216 \
    #-x NCCL_LL_THRESHOLD=16 \
#HOROVOD_TIMELINE=./logs/profile-timeline-${dnn}.json.log HOROVOD_TIMELINE_MARK_CYCLES=1 
#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster-nv$nworkers -bind-to none -map-by slot \
#    -x LD_LIBRARY_PATH -x PATH \
#    -mca pml ob1 -mca btl openib \
#    -mca btl_openib_allow_ib 1\
#    -x NCCL_DEBUG=INFO  \
#    -x NCCL_IB_DISABLE=0 \
#    -x NCCL_BUFFSIZE=16777216 \
#    -x NCCL_LL_THRESHOLD=16 \
#    -x HOROVOD_MPI_THREADS_DISABLE=1 \
#    $PY horovod_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --threshold $threshold --saved-dir $GRADSPATH --momentum-correction $momentum_correction
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster-nv$nworkers -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include eth0 \
    -x NCCL_DEBUG=INFO  \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_IB_DISABLE=1 \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -x NCCL_BUFFSIZE=16777216 \
    -x NCCL_LL_THRESHOLD=16 \
    $PY horovod_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --threshold $threshold --saved-dir $GRADSPATH  --momentum-correction $momentum_correction


# 10GbE configurations
#    -mca pml ob1 -mca btl ^openib \
#    -mca btl_tcp_if_include eth0 \
#    -x NCCL_DEBUG=INFO  \
#    -x NCCL_SOCKET_IFNAME=eth0 \
#    -x NCCL_IB_DISABLE=1 \
#    $PY horovod_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --threshold $threshold --saved-dir $GRADSPATH 

# 56GbIB configurations
#    -mca pml ob1 -mca btl openib \
#    -mca btl_openib_allow_ib 1\
#    -x NCCL_DEBUG=INFO  \
#    -x NCCL_IB_DISABLE=0 \

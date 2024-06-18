#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-8}"
density="${density:-1.0}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"
nwpernode=4
nstepsupdate=1
overlap_scalar=2
strategy='ties'
nsteps_localsgd=20
optimizer_name='SGD'
alg='seq'
PY=~/miniconda3/envs/DDP/bin/python3

MPIPATH=~/miniconda3/envs/DDP/

GRADSPATH=./logs/tzc

$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
    --mca pml ob1 --mca btl self,tcp,vader --mca btl_tcp_if_include bond0 \
    -x LD_LIBRARY_PATH \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=bond0 \
    -x NCCL_TREE_THRESHOLD=0 \
    $PY horovod_trainer.py \
        --alg $alg \
        --optimizer_name $optimizer_name \
        --nsteps_localsgd $nsteps_localsgd \
        --strategy $strategy \
        --overlap_scalar $overlap_scalar \
        --dnn $dnn \
        --dataset $dataset \
        --max-epochs $max_epochs \
        --batch-size $batch_size \
        --nworkers $nworkers \
        --data-dir $data_dir \
        --lr $lr \
        --nsteps-update $nstepsupdate \
        --nwpernode $nwpernode \
        --density $density \
        --compressor $compressor \
        --threshold $threshold \
        --saved-dir $GRADSPATH \
        --momentum-correction $momentum_correction

# $MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster${nworkers} -bind-to none -map-by slot \
#     --mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 \
#     -mca btl_tcp_if_include ib0 \
#     -x LD_LIBRARY_PATH  \
#     -x NCCL_IB_DISABLE=0 \
#     -x NCCL_DEBUG=INFO \
#     -x NCCL_SOCKET_IFNAME=ib0 \
#     -x NCCL_TREE_THRESHOLD=0 \
#     $PY horovod_trainer.py \
#         --alg $alg \
#         --optimizer_name $optimizer_name \
#         --nsteps_localsgd $nsteps_localsgd \
#         --strategy $strategy \
#         --overlap_scalar $overlap_scalar \
#         --dnn $dnn \
#         --dataset $dataset \
#         --max-epochs $max_epochs \
#         --batch-size $batch_size \
#         --nworkers $nworkers \
#         --data-dir $data_dir \
#         --lr $lr \
#         --nsteps-update $nstepsupdate \
#         --nwpernode $nwpernode \
#         --density $density \
#         --compressor $compressor \
#         --threshold $threshold \
#         --saved-dir $GRADSPATH \
#         --momentum-correction $momentum_correction
#!/bin/bash

# Set Python and script environment
#source ../configs/envs.conf
directory=`pwd`
script="${script:-dist_trainer.py}"  # Assuming this is the PyTorch distributed training script
params="${params:-}"
echo "launch dir: $directory"

# Horovod-specific configurations commented out, adjust or remove if unnecessary for PyTorch
#export HOROVOD_WITH_MPI=1
#export HOROVOD_WITH_GLOO=1

hosts=('gpu23' 'gpu22' 'gpu24')
total_host=${#hosts[@]}
# Model and training configurations
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-1.0}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"

# PyTorch Distributed settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-3}"
node_rank="${node_rank:-1}"
node_rank=$(expr $node_rank - 1)  # Adjust for zero-based indexing
if [ $(expr $node_rank + $node_count) -gt $total_host ] || [ $node_rank -lt 0 ]; then
    echo "Required nodes are out of the range: from gpu1 to gpu$total_host"
    exit 0
fi
master_host=${hosts[$node_rank]}

# Training settings
nwpernode=4
nstepsupdate=1
overlap_scalar=2
strategy='average'
nsteps_localsgd=20
optimizer_name='SGD'
alg='localsgd'
PY=~/miniconda3/envs/DDP/bin/python3
GRADSPATH=./logs/tzc

# Loop to launch training on each node
i=0
while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    args="$PY -m torch.distributed.run --nproc_per_node=$ngpu_per_node --nnodes=$node_count --node_rank=$i --master_addr=$master_host --master_port=12345 $script \
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
        --momentum-correction $momentum_correction"
    echo "$host: $args"
    cmd="cd $directory; $args"
    if [ $(expr $i + 1) -eq $node_count ]; then
        ssh $host $cmd   # return until finished or interrupted
    else
        ssh $host $cmd & # return immediately
    fi
    node_rank=$(expr $node_rank + 1)
    i=$(expr $i + 1)
done

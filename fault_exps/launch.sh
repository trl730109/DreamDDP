#!/bin/bash

# Set Python and script environment
#source ../configs/envs.conf
directory=`pwd`
script="${script:-fault_dist_trainer.py}"  # Assuming this is the PyTorch distributed training script
params="${params:-}"
echo "launch dir: $directory"

# Horovod-specific configurations commented out, adjust or remove if unnecessary for PyTorch
#export HOROVOD_WITH_MPI=1
#export HOROVOD_WITH_GLOO=1
total_host=1
# hosts=('gpu23')
# Model and training configurations
dnn="${dnn:-resnet18}"
# source fault_exps/model_configs/$dnn.conf

cluster_name="${cluster_name:-localhost}"
source fault_exps/env_configs/$cluster_name.sh


nworkers="${nworkers:-4}"
density="${density:-1.0}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"

# PyTorch Distributed settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"
node_rank=$(expr $node_rank - 1)  # Adjust for zero-based indexing
if [ $(expr $node_rank + $node_count) -gt $total_host ] || [ $node_rank -lt 0 ]; then
    echo "Required nodes are out of the range: from gpu1 to gpu$total_host"
    exit 0
fi
master_host=${hosts[$node_rank]}

dnn="${dnn:-resnet18}"
lr="${lr:-0.1}"
batch_size="${batch_size:-128}"

max_epochs="${max_epochs:-181}"


# Training settings
nwpernode="${nwpernode:-$ngpu_per_node}"
nstepsupdate=1
overlap_scalar=2
strategy='average'
nsteps_localsgd="${nsteps_localsgd:-20}"
optimizer_name="${optimizer_name:-SGD}"
sync="${sync:-avg}"
alg="${alg:-sgd}"
# PY=~/miniconda3/envs/DDP/bin/python3
GRADSPATH=./logs/tzc

dataset="${dataset:-cifar10}"
data_dir="${data_dir:-/home/comp/amelieczhou/datasets/cifar10}"

# exp_name="${exp_name:-default}"

# Loop to launch training on each node
i=0

project_name=DDP-Train

nworkers=$(expr $nwpernode \* $node_count)


exp_name=${alg}-noi${add_noise}-${dnn}-nw${nworkers}-${optimizer_name}-LG${nsteps_localsgd}-lr${lr}-bs${batch_size}
echo "exp name is $exp_name !"

while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    args="$PY -m torch.distributed.run --nproc_per_node=$ngpu_per_node --nnodes=$node_count --node_rank=$i --master_addr=$master_host --master_port=23456 $script \
        --alg $alg \
        --exp_name $exp_name \
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
        --momentum-correction $momentum_correction \
        --sync $sync \
        --add_noise $add_noise \
        --gaussian_mu $gaussian_mu \
        --gaussian_std $gaussian_std \
        --wandb_entity $wandb_entity --project_name $project_name --enable_wandb $enable_wandb --wandb_offline $wandb_offline \
        --wandb_key $wandb_key \
        --exp_name $exp_name "
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










#!/bin/bash

# Set Python and script environment
directory=$(pwd)
script="${script:-dist_trainer.py}"  # Assuming this is the PyTorch distributed training script
params="${params:-}"
echo "launch dir: $directory"

total_host=${#hosts[@]}
# hosts=('gpu23')
# Model and training configurations
dnn="${dnn:-resnet18}"
source train_exps/env_configs/$cluster_name.sh
echo "dataset dir: $data_dir"

export NCCL_DEBUG=TRACE

# Optionally, focus on socket information
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eno0
nworkers="${nworkers:-4}"
density="${density:-1.0}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"

# PyTorch Distributed settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count="${node_count:-1}"
node_rank="${node_rank:-1}"
echo "node_count: $node_count"
node_rank=$(expr $node_rank - 1)  # Adjust for zero-based indexing
if [ $(expr $node_rank + $node_count) -gt $total_host ] || [ $node_rank -lt 0 ]; then
    echo "node_rank: $node_rank"
    echo "node_count: $node_count"
    echo "Required nodes are out of the range: from gpu1 to gpu$total_host"
    exit 0
fi
master_host=${hosts[$node_rank]}
wandb_key="${wandb_key:-None}"
# Training settings
nwpernode=4
nstepsupdate=1
interface="${interface:-eno0}"
overlap_scalar=2
strategy='average'
nsteps_localsgd="${nsteps_localsgd:-20}"
optimizer_name="${optimizer_name:-SGD}"
sync="${sync:-avg}"
alg="${alg:-sgd}"
GRADSPATH=./logs/tzc
lr_decay="${lr_decay:-None}"
dataset="${dataset:-cifar10}"
data_dir="${data_dir:-/home/comp/amelieczhou/datasets/cifar10}"
group_num="${group_num:-6}"
if [ "$interface" = "eno0" ]; then
    bandwidth="1G"
elif [ "$interface" = "ens5f0" ]; then
    bandwidth="10G"
else
    bandwidth="100G"
fi

exp_name="${exp_name:-default}"
extra_name="${extra_name:- }"
if [ "$alg" = "pipe_seq_localsgd" ]; then
    exp_name="${extra_name}-${alg}-${sync}-${dnn}-${dataset}-${nsteps_localsgd}-${bandwidth}-lr${lr}-lr_decay${lr_decay}-nodes${total_host}-nworkers${nworkers}"
elif [ "$alg" = "localsgd" ]; then
    exp_name="${extra_name}-${alg}-${dnn}-${dataset}-${nsteps_localsgd}-${bandwidth}-lr${lr}-lr_decay${lr_decay}-nodes${total_host}-nworkers${nworkers}"
elif [ "$alg" = "transformer_localsgd" ]; then
    exp_name="${extra_name}-${alg}-${dnn}-${dataset}-${nsteps_localsgd}-${bandwidth}-lr${lr}-lr_decay${lr_decay}-nodes${total_host}-nworkers${nworkers}"
    echo "Exp name: $exp_name"
elif [ "$alg" = "full_pipe_seq" ]; then
    exp_name="${extra_name}-${alg}_${group_num}-${dnn}-${dataset}-${nsteps_localsgd}-${bandwidth}-lr${lr}-lr_decay${lr_decay}-nodes${total_host}-nworkers${nworkers}"
    echo "Exp name: $exp_name"
else
    exp_name="${extra_name}-${alg}-${dnn}-${dataset}-nstepsupdate${nstepsupdate}-${bandwidth}-lr${lr}-lr_decay${lr_decay}-nodes${total_host}-nworkers${nworkers}"
    echo "Exp name: $exp_name"
fi

if [ -z "$exp_name" ]; then
    echo "Error: exp_name is empty."
    exit 1
fi
# Loop to launch training on each node
i=0

project_name=DDP-Train

while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    echo "Entering node: $host"
    args="$PY -m torch.distributed.run --nproc_per_node=$ngpu_per_node --nnodes=$node_count --node_rank=$i --master_addr=$master_host --master_port=2384 $script \
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
        --lr_decay $lr_decay \
        --group_num $group_num \
        --nsteps-update $nstepsupdate \
        --nwpernode $nwpernode \
        --density $density \
        --compressor $compressor \
        --interface $interface \
        --threshold $threshold \
        --saved-dir $GRADSPATH \
        --momentum-correction $momentum_correction \
        --wandb_entity $wandb_entity --project_name $project_name --enable_wandb $enable_wandb --wandb_offline $wandb_offline \
        --wandb_key $wandb_key"
    echo "$host: $args"
    cmd="cd $directory; $args"
    if [ $(expr $i + 1) -eq $node_count ]; then
        ssh $host $cmd   # return until finished or interrupted
    else
        ssh $host $cmd &
    fi
    node_rank=$(expr $node_rank + 1)
    i=$(expr $i + 1)
done

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
total_host=16
# hosts=('gpu23')
# Model and training configurations
dnn="${dnn:-resnet18}"
# source fault_exps/model_configs/$dnn.conf
echo "cluster name: $cluster_name"
source fault_exps/env_configs/$cluster_name.sh
echo "dataset dir: $data_dir"
echo "model_dir: $model_dir"

pre_cmd="${pre_cmd:-}"
echo "pre_cmd: $pre_cmd"

# export NCCL_DEBUG=TRACE


export CUDA_VISIBLE_DEVICES=1,2,3.4



nworkers="${nworkers:-4}"
density="${density:-1.0}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"

# PyTorch Distributed settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count=${#hosts[@]}
node_rank=1
node_rank=$(expr $node_rank - 1)  # Adjust for zero-based indexing
if [ $(expr $node_rank + $node_count) -gt $total_host ] || [ $node_rank -lt 0 ]; then
    echo "node_rank + node_count is $(expr $node_rank + $node_count) Required nodes are out of the range: from gpu1 to gpu$total_host"
    exit 0
fi
master_host=${hosts[$node_rank]}

dnn="${dnn:-resnet18}"
lr="${lr:-0.1}"
lr_decay="${lr_decay:-general}"
weight_decay="${weight_decay:-0.0001}"
adam_beta1="${adam_beta1:-0.9}"
adam_beta2="${adam_beta2:-0.999}"
batch_size="${batch_size:-128}"

max_epochs="${max_epochs:-181}"


# Training settings
nwpernode="${nwpernode:-$ngpu_per_node}"
nstepsupdate="${nstepsupdate:-1}"
overlap_scalar=2
strategy='average'
nsteps_localsgd="${nsteps_localsgd:-20}"
optimizer_name="${optimizer_name:-SGD}"
sync="${sync:-avg}"
alg="${alg:-sgd}"
PY=/home/tangzhenheng/anaconda3/envs/fusionai/bin/python
GRADSPATH=./logs/tzc

dataset="${dataset:-cifar10}"
data_dir="${data_dir:-/home/comp/amelieczhou/datasets/cifar10}"
model_dir="${model_dir:-/mnt/raid/gpt2}"
load_pretrain="${load_pretrain:-False}"
training_type="${training_type:-pretrain}"
finetune_type="${finetune_type:-lora}"
peft_lora_r="${peft_lora_r:-8}"
peft_lora_alpha="${peft_lora_alpha:-16}"

# exp_name="${exp_name:-default}"

# Loop to launch training on each node
i=0



# noise_type="${noise_type:-}"burst
noise_type="${noise_type:-fix}"
burst_freq="${burst_freq:-10}"
burst_magnitude="${burst_magnitude:-1.0}"

project_name=DDP-Train

nworkers=$(expr $nwpernode \* $node_count)

extra_name="${extra_name:-}"

exp_name=${alg}-noi${add_noise}-t${noise_type}-${dnn}-${finetune_type}-nw${nworkers}-${optimizer_name}-LG${nsteps_localsgd}-lr${lr}-bs${batch_size}-${extra_name}
echo "exp name is $exp_name !"

nsteps_param_sync=${nsteps_param_sync:-20}
check_param_diversity=${check_param_diversity:-True}
nsteps_param_diversity=${nsteps_param_diversity:-5}
param_sync=${param_sync:-"fix"}

master_port=${master_port:-23456}

wandb_offline=False

while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    args="$pre_cmd $PY -m torch.distributed.run --nproc_per_node=$ngpu_per_node --nnodes=$node_count --node_rank=$i --master_addr=$master_host --master_port=$master_port $script \
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
        --model_dir $model_dir \
        --load_pretrain $load_pretrain \
        --training_type $training_type \
        --finetune_type $finetune_type \
        --peft_lora_r $peft_lora_r \
        --peft_lora_alpha $peft_lora_alpha \
        --lr $lr \
        --lr_decay $lr_decay \
        --weight_decay $weight_decay \
        --adam_beta1 $adam_beta1 \
        --adam_beta2 $adam_beta2 \
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
        --noise_type=$noise_type \
        --burst_freq=$burst_freq \
        --burst_magnitude=$burst_magnitude \
        --nsteps_param_sync $nsteps_param_sync \
        --check_param_diversity $check_param_diversity \
        --nsteps_param_diversity $nsteps_param_diversity \
        --param_sync $param_sync \
        --param_sync_async_op $param_sync_async_op \
        --wandb_entity $wandb_entity --project_name $project_name --enable_wandb $enable_wandb --wandb_offline $wandb_offline \
        --wandb_key $wandb_key \
        --exp_name $exp_name "
    echo "$host: $args"
    cmd="cd $directory; $args"
    if [ $(expr $i + 1) -eq $node_count ]; then
        ssh $host $cmd # return until finished or interrupted
    else
        ssh $host $cmd >> /dev/null & # return immediately
    fi
    node_rank=$(expr $node_rank + 1)
    i=$(expr $i + 1)
done










#!/bin/bash

# 本脚本由 transformer_sgd.sh 拷贝修改而来，
# 目的：在本机 localhost 上直接用 torch.distributed.run 起进程，不经过 ssh/launch_transformer_A6000.sh。

# ===== 与 launch_transformer_A6000.sh / env_configs/A6000.sh 对齐的环境部分 =====
cluster_name=A6000
dataset='wikitext2'
dnn='Qwen2.5-1.5B'
PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"
# 复用统一的环境与路径配置（PY、data_dir、model_dir）
source "$(dirname "$0")/env_configs/A6000.sh"

# ===== 本地实验超参数 =====
lr=0.0001
batch_size=1
alg='transformer_sgd'

interface=eth0

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"

optimizer_name=Adam

max_epochs=1
extra_name='Qwen2.5-1.5B'

enable_wandb=false
wandb_offline=true
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5

# 单机本地跑，直接用 localhost，不再 ssh
master_addr="127.0.0.1"
master_port=2260
node_count=1
nwpernode=8
nworkers=$((nwpernode * node_count))
ngpu_per_node=$nwpernode
nsteps_localsgd=10

adam_beta1=0.9
lr=0.0001
weight_decay=0.0001
lr_decay='fixed'

# 其它与 launch_transformer_A6000.sh 中一致的参数
group_num=6
strategy='average'
overlap_scalar=2
density=1.0
compressor='none'
threshold=524288000
saved_dir="./logs/tzc"
enlarge=false
momentum_correction=0
project_name="DDP-Train"

# 构造一个简单的实验名（不依赖外部脚本）
exp_name="${extra_name}-${alg}-${dnn}-${dataset}-nstepsupdate1-100G-lr${lr}-lr_decay${lr_decay}-nodes${node_count}-nworkers${nworkers}"
echo "Exp name: $exp_name"

echo "launch dir: $(pwd)"
echo "dataset dir: $data_dir"
echo "model_dir: $model_dir"
echo "node_count: $node_count"

NPROC_PER_NODE=$nwpernode

cmd="$pre_cmd  $PY -m torch.distributed.run \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$node_count \
  --node_rank=0 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  dist_trainer_transformer.py \
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
    --lr $lr \
    --lr_decay $lr_decay \
    --group_num $group_num \
    --nsteps-update 1 \
    --nwpernode $nwpernode \
    --density $density \
    --compressor $compressor \
    --interface $interface \
    --threshold $threshold \
    --saved-dir $saved_dir \
    --enlarge $enlarge \
    --check_param_diversity $check_param_diversity \
    --nsteps_param_diversity $nsteps_param_diversity \
    --momentum-correction $momentum_correction \
    --dist_backend gloo \
    --wandb_entity $wandb_entity --project_name $project_name --enable_wandb $enable_wandb --wandb_offline $wandb_offline \
    --wandb_key $wandb_key"

echo "Running locally on localhost with command:"
echo "$cmd"

eval "$cmd"
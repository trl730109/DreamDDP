#!/bin/bash
# ResNet profile: 2 算法 × 2 模型 (resnet18, resnet50) = 4 runs
# 每次 run 前显式设置相关变量并递增 master_port，避免实验间互相污染。

PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"
script="dist_trainer.py"
pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"

lr=0.1
batch_size=128
max_epochs=5
optimizer_name=SGD
interface=eth0

enable_wandb=False
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5

profile=True
cluster_name=A6000
hosts=('10.244.5.206')
ports=(22)

node_count=${#hosts[@]}
nwpernode=8
nworkers=$((nwpernode * node_count))
ngpu_per_node=$nwpernode
nsteps_localsgd=10
nstepsupdate=1

node_rank=1
lr_decay='exp'
extra_name="${node_count}Nodes"

# ========== resnet18 + sgd ---
master_port=2778
alg='sgd'
dnn=resnet18
dataset=cifar10
source train_exps/launch_mul.sh
master_port=$((master_port + 1))

# ========== resnet50 + sgd ---
alg='sgd'
dnn=resnet50
dataset=cifar100
source train_exps/launch_mul.sh
master_port=$((master_port + 1))

# ========== resnet18 + localsgd ---
alg='localsgd'
sync_momentum=true
dnn=resnet18
dataset=cifar10
source train_exps/launch_mul.sh
master_port=$((master_port + 1))

# ========== resnet50 + localsgd ---
alg='localsgd'
sync_momentum=true
dnn=resnet50
dataset=cifar100
source train_exps/launch_mul.sh

#!/bin/bash
# Unified transformer profile: 3 models (2 full + 1 lora) × 2 algs (transformer_sgd + transformer_localsgd) = 6 runs
# 每次 run 前显式设置所有相关变量并递增 master_port，避免实验间互相污染。

lr=0.0001
batch_size=1
dataset='wikitext2'
max_epochs=3

data_dir="/mnt/raid/tangzichen/wikitext2"
interface=eth0
PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"

optimizer_name=Adam
enable_wandb=false
wandb_offline=true
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5

cluster_name=A6000


hosts=('10.244.1.115' '10.244.3.176')
ports=(22 22)



profile=True

master_port=2778
node_count=${#hosts[@]}
nwpernode=8
nworkers=$((nwpernode * node_count))
ngpu_per_node=$nwpernode
nsteps_localsgd=10

adam_beta1=0.9
weight_decay=0.0001
node_rank=1
lr_decay='fixed'

# ========== 先跑 4 个 full（2 模型 × 2 算法）==========

# --- full: gpt2 + transformer_sgd ---
alg='transformer_sgd'
dnn=gpt2
extra_name="${dnn}"
model_dir="/workspace/models/${dnn}"
finetune_type=full
peft_lora_r=8
peft_lora_alpha=16
source train_exps/launch_transformer_A6000.sh
master_port=$((master_port + 1))

# --- full: llama2-124M + transformer_sgd ---
alg='transformer_sgd'
dnn=llama2-124M
extra_name="${dnn}"
model_dir="/workspace/models/${dnn}"
finetune_type=full
peft_lora_r=8
peft_lora_alpha=16
source train_exps/launch_transformer_A6000.sh
master_port=$((master_port + 1))

# --- full: gpt2 + transformer_localsgd ---
alg='transformer_localsgd'
dnn=gpt2
extra_name="${dnn}"
model_dir="/workspace/models/${dnn}"
finetune_type=full
peft_lora_r=8
peft_lora_alpha=16
source train_exps/launch_transformer_A6000.sh
master_port=$((master_port + 1))

# --- full: llama2-124M + transformer_localsgd ---
alg='transformer_localsgd'
dnn=llama2-124M
extra_name="${dnn}"
model_dir="/workspace/models/${dnn}"
finetune_type=full
peft_lora_r=8
peft_lora_alpha=16
source train_exps/launch_transformer_A6000.sh
master_port=$((master_port + 1))

# ========== 最后跑 2 个 lora（1 模型 × 2 算法）==========

# --- lora: Qwen2.5-7B + transformer_sgd（可按需改 model_dir）---
alg='transformer_sgd'
dnn=Qwen2.5-7B
extra_name="${dnn}-lora"
model_dir="/workspace/models/${dnn}"
finetune_type=lora
peft_lora_r=16
peft_lora_alpha=32
source train_exps/launch_transformer_A6000.sh
master_port=$((master_port + 1))

# --- lora: Qwen2.5-7B + transformer_localsgd ---
alg='transformer_localsgd'
dnn=Qwen2.5-7B
extra_name="${dnn}-lora"
model_dir="/workspace/models/${dnn}"
finetune_type=lora
peft_lora_r=16
peft_lora_alpha=32
source train_exps/launch_transformer_A6000.sh

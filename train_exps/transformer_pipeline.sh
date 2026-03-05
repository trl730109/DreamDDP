#!/bin/bash
# 合并的 transformer pipeline: 先 profile，然后运行调度，最后运行实际训练
# 顺序：profile -> Scheduling/run.sh -> 实际训练
# Usage: bash transformer_pipeline.sh [all|train]
#   all: 执行所有步骤（包括 profile）
#   train: 跳过 profile，只执行 scheduling 和训练

MODE=${1:-all}

lr=0.0001
batch_size=1
dataset='wikitext2'

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

# hosts=('10.244.1.115' '10.244.3.176' '10.244.10.55' '10.244.4.104')
# ports=(22 22 22 22)

# hosts=('10.244.9.73' '10.244.19.3')
# ports=(22 22)

hosts=('10.244.19.3' '10.244.3.185')
ports=(22 22)

master_port=3124
node_count=${#hosts[@]}
nwpernode=8
nworkers=$((nwpernode * node_count))
ngpu_per_node=$nwpernode
nsteps_localsgd=10

adam_beta1=0.9
weight_decay=0.0001
node_rank=1
lr_decay='fixed'
enlarge=false

BP_MULTIPLIER=1
COMM_MULTIPLIER=1

# # 定义模型列表
# declare -a dnn_list=(
#     "gpt2"
#     "llama2-124M"
#     "Qwen2.5-7B"
# )

declare -a dnn_list=(
    "gpt2"
)

bandwidth="10gbit"
max_epochs=3
# ========== Step 1: Profile ==========
if [ "$MODE" = "all" ]; then
    echo "========== Starting Profile =========="
    profile=True
    enable_wandb=false

    for dnn in "${dnn_list[@]}"; do
        # Set parameters based on model
        if [ "$dnn" = "Qwen2.5-7B" ]; then
            finetune_type=lora
            peft_lora_r=16
            peft_lora_alpha=32
            extra_name="${dnn}-lora"
        else
            finetune_type=full
            peft_lora_r=8
            peft_lora_alpha=16
            extra_name="${dnn}"
        fi
        model_dir="/workspace/models/${dnn}"
        
        # Profile transformer_sgd (full-precision)
        alg='transformer_sgd'
        source train_exps/launch_transformer_A6000.sh
        master_port=$((master_port + 1))
        
        # Profile transformer_localsgd (low-precision)
        alg='transformer_localsgd'
        source train_exps/launch_transformer_A6000.sh
        master_port=$((master_port + 1))
    done
fi


echo "========== Step 2: Scheduling Generation Starts =========="

time_base="/workspace/tzc/DreamDDP/time"
worker_path="${nworkers}"

for dnn in "${dnn_list[@]}"; do
    echo "--- Scheduling Generation: ${dnn} ---"
    python3 Scheduling/dreamddp_scheduling.py ${time_base}/${dnn}/${worker_path}/${bandwidth} --H 10 --bp_multiplier ${BP_MULTIPLIER} --comm_multiplier ${COMM_MULTIPLIER}
done

echo "========== Scheduling Generation Completed =========="

profile=False
enable_wandb=true
max_epochs=3

for dnn in "${dnn_list[@]}"; do
    # Set parameters based on model
    if [ "$dnn" = "Qwen2.5-7B" ]; then
        finetune_type=lora
        peft_lora_r=16
        peft_lora_alpha=32
        extra_name="${dnn}-lora"
    else
        finetune_type=full
        peft_lora_r=8
        peft_lora_alpha=16
        extra_name="${dnn}"
    fi
    
    # # Train transformer_sgd (full-precision)
    # alg='transformer_sgd'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))
    
    # Train transformer_localsgd (low-precision)
    alg='transformer_localsgd'
    source train_exps/launch_transformer_A6000.sh
    master_port=$((master_port + 1))
    
#     # Train transformer_dream_ddp_optimized (optimized)
#     alg='transformer_dream_ddp'
#     source train_exps/launch_transformer_A6000.sh
#     master_port=$((master_port + 1))

#     # Train transformer_dream_ddp_optimized (optimized)
#     alg='transformer_dream_ddp_optimized'
#     source train_exps/launch_transformer_A6000.sh
#     master_port=$((master_port + 1))
done

#!/bin/bash
MODE=${1:-all}

lr=0.0001
batch_size=1
dataset='wikitext2'

interface=eth0
PY="${PY:-/workspace/pretrain/miniconda3/envs/ddp_moe/bin/python}"
pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"

optimizer_name=Adam
enable_wandb=false
wandb_offline=true
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5

cluster_name=A6000


# hosts=('10.244.3.188' '10.244.4.109')
# ports=(22 22)

hosts=('10.244.13.201')
ports=(22)
nwpernode=4   # number of GPUs per node

master_port=3357
node_count=${#hosts[@]}
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
declare -a dnn_list=(
    "gpt2"
    # "llama2-124M"
    # "Qwen2.5-7B"
    # "Qwen2.5-MoE"
    # "granite-MoE"
)

load_pretrain=true
# load_quantization=4bit

bandwidth="1gbit"
# bandwidth="1Gbps"
max_epochs=1
# ========== Step 0: SSH Check ==========
# Verify all training nodes are reachable before starting.
# If this fails, run bash ssh_conf.sh from your local machine first.
echo "========== Step 0: SSH Connectivity Check =========="
ssh_ok=true
for i in "${!hosts[@]}"; do
    h="${hosts[$i]}"
    p="${ports[$i]}"
    if ssh -p "$p" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "root@$h" "exit" 2>/dev/null; then
        echo "  [OK]   root@$h:$p"
    else
        echo "  [FAIL] root@$h:$p"
        ssh_ok=false
    fi
done
if [ "$ssh_ok" = false ]; then
    echo "ERROR: Some nodes unreachable. Run: bash ssh_conf.sh"
    exit 1
fi
echo ""

# ========== Step 1: Profile ==========
if [ "$MODE" = "all" ]; then
    echo "========== Starting Profile =========="
    profile=True
    enable_wandb=false

    for dnn in "${dnn_list[@]}"; do
        # Set parameters based on model
        if [ "$dnn" = "Qwen2.5-7B" ] || [ "$dnn" = "Qwen2.5-MoE" ]; then
            finetune_type=lora
            peft_lora_r=8
            peft_lora_alpha=16
            extra_name="${dnn}-lora"
        else
            finetune_type=full
            peft_lora_r=8
            peft_lora_alpha=16
            extra_name="${dnn}"
        fi
        # model_dir="/workspace/models/${dnn}"
        
        # # Profile transformer_sgd (full-precision)
        # alg='transformer_sgd'
        # source train_exps/launch_transformer_A6000.sh
        # master_port=$((master_port + 1))
        
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
enable_wandb=false
max_epochs=1

for dnn in "${dnn_list[@]}"; do
    # Set time stamp
    time_stamp_dnn="${dnn}-$(date +%Y%m%d_%H%M%S)"
    time_stamp=$time_stamp_dnn

    profiler_trace=False
    cpu_clock=True


    if [ "$dnn" = "Qwen2.5-7B" ] || [ "$dnn" = "Qwen2.5-MoE" ]; then
        finetune_type=lora
        peft_lora_r=8
        peft_lora_alpha=16
        extra_name="${dnn}-lora"
    else
        finetune_type=full
        peft_lora_r=8
        peft_lora_alpha=16
        extra_name="${dnn}"
    fi
    
    # # Train transformer_sgd
    # alg='transformer_sgd'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))

    # Train transformer_pipe_sgd
    # alg='transformer_pipe_sgd'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))
    
    # # Train transformer_localsgd
    # alg='transformer_localsgd'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))
    
    # # Train transformer_dream_ddp
    # alg='transformer_dream_ddp'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))

    # # Calculate speedup of dreamddp relative to pipe_sgd and localsgd
    # $PY train_exps/speedup_stats.py --time_stamp "$time_stamp" --dnn "$dnn" --nworkers "$nworkers" --bandwidth "$bandwidth"

    
done

#!/bin/bash
MODE=${1:-all}

# ============================================================
# Cluster Config  <-- edit this when switching clusters
# ============================================================
# cluster_name=A6000

cluster_name=your_cluster_name
# PY is set in train_exps/env_configs/env.sh based on cluster_name

# Node IPs and SSH ports (internal network)
nwpernode=4    # GPUs per node
hosts=('node_ip_1' 'node_ip_2')
ports=(22 22)
# Examples
# hosts=('10.244.4.118' '10.244.3.196' '10.244.5.219' '10.244.1.135')
# ports=(22 22 22 22)

master_port=3357
bandwidth="1gbit"    # used for scheduling and (optionally) tc bandwidth shaping
enable_tc=false      # set true to throttle inter-node bandwidth via tc (requires cap_net_admin)

# WandB
enable_wandb=false  # set true to enable WandB logging
wandb_offline=true  # set true to run WandB in offline mode
wandb_entity=your_wandb_entity
wandb_key=your_wandb_key

# ============================================================
# Model List  <-- uncomment models to run
# ============================================================
declare -a dnn_list=(
    "gpt2"
    # "llama2-124M"
    # "Qwen2.5-7B"
    # "Qwen2.5-MoE"
    # "granite-MoE"
)

# ============================================================
# Training Hyperparameters
# ============================================================
lr=0.0001
batch_size=1
dataset='wikitext2'
max_epochs=1
optimizer_name=Adam
adam_beta1=0.9
weight_decay=0.0001
lr_decay='fixed'
nsteps_localsgd=10
load_pretrain=true
# load_quantization=4bit

# Scheduling multipliers
BP_MULTIPLIER=1
COMM_MULTIPLIER=1

# Misc
interface=eth0
pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"
check_param_diversity=false
nsteps_param_diversity=5
enlarge=false
node_rank=1

# Derived
node_count=${#hosts[@]}
nworkers=$((nwpernode * node_count))
ngpu_per_node=$nwpernode

# ============================================================
# Step 0: SSH Check
# ============================================================
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

# ============================================================
# Helper: set finetune params per model
# ============================================================
set_model_params() {
    local dnn=$1
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
}

# ============================================================
# Step 1: Profile
# ============================================================
if [ "$MODE" = "all" ]; then
    echo "========== Step 1: Profile =========="
    profile=True
    enable_wandb=false

    for dnn in "${dnn_list[@]}"; do
        set_model_params "$dnn"

        # alg='transformer_sgd'
        # source train_exps/launch_transformer_A6000.sh
        # master_port=$((master_port + 1))

        alg='transformer_localsgd'
        source train_exps/launch_transformer_A6000.sh
        master_port=$((master_port + 1))
    done
fi

# ============================================================
# Step 2: Scheduling
# ============================================================
echo "========== Step 2: Scheduling =========="
time_base="/workspace/tzc/DreamDDP/time"
for dnn in "${dnn_list[@]}"; do
    echo "--- ${dnn} ---"
    python3 Scheduling/dreamddp_scheduling.py ${time_base}/${dnn}/${nworkers}/${bandwidth} \
        --H 10 --bp_multiplier ${BP_MULTIPLIER} --comm_multiplier ${COMM_MULTIPLIER}
done

# ============================================================
# Step 3: Training  <-- uncomment algorithms to run
# ============================================================
echo "========== Step 3: Training =========="
profile=False
enable_wandb=false
max_epochs=1

for dnn in "${dnn_list[@]}"; do
    set_model_params "$dnn"
    time_stamp="${dnn}-$(date +%Y%m%d_%H%M%S)"
    profiler_trace=False
    cpu_clock=True

    # alg='transformer_sgd'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))

    # alg='transformer_pipe_sgd'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))

    # alg='transformer_localsgd'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))

    # alg='transformer_dream_ddp'
    # source train_exps/launch_transformer_A6000.sh
    # master_port=$((master_port + 1))

    # $PY train_exps/speedup_stats.py --time_stamp "$time_stamp" --dnn "$dnn" --nworkers "$nworkers" --bandwidth "$bandwidth"
done

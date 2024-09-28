

master_port=12345

alg=sgd_with_sync_all
# alg=sgd
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=Adam
lr=0.01
lr_decay=fixed
dnn=gpt2
dataset=wikitext2

training_type="${training_type:-pretrain}"
finetune_type="${finetune_type:-full}"
peft_lora_r="${peft_lora_r:-8}"
peft_lora_alpha="${peft_lora_alpha:-16}"

batch_size=4
max_epochs=5

add_noise=True
# add_noise=False

# enable_wandb=False
enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
exp_name=$exp_name
# cluster_name=gpuhome
# cluster_name=scigpu
# hosts=('scigpu14')
# cluster_name=A6000
cluster_name=GZA6000
# hosts=('ibgpu3')
hosts=('localhost')
# ports=(31949)



# node_count=${#ports[@]}
# nworkers=$((8 * node_count))
# nwpernode=8
# ngpu_per_node=$nwpernode
# extra_name="${node_count}Nodes"

nstepsupdate=4
adam_beta1=0.9
adam_beta2=0.95
# lr=0.0001
weight_decay=0.0001

lr_decay='fixed'

# cluster_name=esetstore
# hosts=('gpu3')
param_sync_async_op=False
# param_sync=detect_base
param_sync=fix

check_param_diversity=False
nsteps_param_diversity=5
nsteps_param_sync=5

add_noise=True

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"
PY="${PY:-/mnt/sdb/tangzhenheng/miniconda3/envs/DDP_Train/bin/python}"
# PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

# model_dir="/data2/share/zhtang/llama-2-7b-hf"
# dnn=gpt2
# model_dir="/data2/share/zhtang/gpt2"


values=(5)
# values=(5 10 50 100)
# values=(10 50)
# nsteps_param_sync=100

for nsteps_param_sync in "${values[@]}"
do
    # gaussian_std=0.001
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

    gaussian_std=0.01
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh

    # gaussian_std=0.1
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

done




























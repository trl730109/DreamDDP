

master_port=12345

# alg=sgd
alg=sgd_with_sync
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=Adam
lr=0.0001
lr_decay=fixed
dnn=gpt2
# dataset=wikitext2
dataset="tatsu-lab/alpaca"
# dataset="vicgalle/alpaca-gpt4"


# tatsu-lab/alpaca


batch_size=8


training_type="${training_type:-finetune}"
# training_type="${training_type:-pretrain}"
# finetune_type="${finetune_type:-full}"
finetune_type="${finetune_type:-lora}"

peft_lora_r="${peft_lora_r:-8}"
peft_lora_alpha="${peft_lora_alpha:-16}"


max_epochs=1

add_noise=True
# add_noise=False

# enable_wandb=False
enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name
# cluster_name=gpuhome
# cluster_name=scigpu
# hosts=('scigpu14')
cluster_name=GZ4090
# cluster_name=A6000
# cluster_name=GZA6000
hosts=('localhost')

# cluster_name=esetstore
# hosts=('gpu3')
param_sync_async_op=False
param_sync=detect_base
# param_sync=fix

check_param_diversity=False
nsteps_param_diversity=5
nsteps_param_sync=5

max_epochs=10

add_noise=True

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"
PY="${PY:-/mnt/sdb/tangzhenheng/miniconda3/envs/DDP_Train/bin/python}"
# PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

# dnn=llama2-124M
# model_dir="/data2/share/zhtang/llama-2-7b-hf"
# dnn=gpt2
model_dir="/data2/share/zhtang/newgpt2/gpt2"


values=(5)
# values=(5 10 50 100)
# values=(10 50)
# nsteps_param_sync=100

for nsteps_param_sync in "${values[@]}"
do
    # gaussian_std=0.0001
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

    # gaussian_std=0.001
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

    gaussian_std=0.01
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh

    # gaussian_std=0.1
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

    # gaussian_std=1.0
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

    # gaussian_std=100.0
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

done
# gaussian_std=10.0
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh


# gaussian_std=100.0
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh


# gaussian_std=1000.0
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh









# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  






























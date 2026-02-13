lr=0.00001
batch_size=1
alg='transformer_sgd'
dataset='wikitext2'

data_dir="/mnt/raid/tangzichen/wikitext2"
interface=eth0
PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"

optimizer_name=Adam
dnn=gpt2

max_epochs=5
# add_noise=True
extra_name="${dnn}"

enable_wandb=false
wandb_offline=true
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name

cluster_name=A6000

hosts=('10.244.5.206')
ports=(22)

profile=True
#
#
master_port=2778
node_count=${#hosts[@]}
nwpernode=8
nworkers=$((nwpernode * node_count))
ngpu_per_node=$nwpernode
nsteps_localsgd=10

# lr_decay=None

adam_beta1=0.9
# lr=0.00001
lr=0.0001

weight_decay=0.0001
node_rank=1
lr_decay='fixed'

dnn=gpt2
source train_exps/launch_transformer_A6000.sh

dnn=llama2-124M
source train_exps/launch_transformer_A6000.sh

# /workspace/DDP-Train/train_exps/launch_transformer_A6000.sh
# dnn=llama2-124M
# model_dir="/data2/share/zhtang/llama-2-7b-hf"
# extra_name='llama2-124M-Notload'
# load_pretrain=False

# weight_decays=(0.0001 0.001 0.01)
# lrs=(0.00001 0.00003 0.0001 0.0003 0.001)
# for weight_decay in "${weight_decays[@]}"
# do
#     for lr in "${lrs[@]}"
#     do
#         # weight_decay=weight_decay
#         # lr=lr
#         source train_exps/debug_launch_transformer.sh
#     done
# done





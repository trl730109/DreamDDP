lr=0.00001
batch_size=4
alg='transformer_sgd'
dataset='wikitext2'

data_dir="/mnt/raid/tangzichen/wikitext2"
# model_dir="/mnt/raid/tangzichen/gpt2"

model_dir="/mnt/raid/tangzichen/bert-base-uncased"
# model_dir="/workspace/models/gpt2"
# "/workspace/models/gpt2"

# data_dir="/home/tangzhenheng/wikitext2"
# model_dir="/home/tangzhenheng/gpt2"
# PY="/mnt/raid/tangzhenheng/anaconda3/bin/python"
PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"
# PY="/home/tangzhenheng/anaconda3/bin/python"


pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"

optimizer_name=Adam
# dnn=llama2-124M
dnn=gpt2

max_epochs=2
# add_noise=True
extra_name='gpt-load-pretrain'

enable_wandb=false
wandb_offline=false
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name
# cluster_name=GZ4090ZHTANG
cluster_name=A6000


# hosts=('10.0.0.20')
# hosts=('30332' '30737')
hosts=("haigpu4")
#
master_port=2280
node_count=${#hosts[@]}
nworkers=$((8 * node_count))
nwpernode=8
ngpu_per_node=$nwpernode
nsteps_localsgd=10

# lr_decay=None

adam_beta1=0.9
# lr=0.00001
lr=0.00001

weight_decay=0.0001

lr_decay='fixed'
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





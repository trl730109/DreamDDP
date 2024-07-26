lr=0.01
batch_size=8
alg='transformer_localsgd'
dataset='wikitext2'


data_dir="/mnt/raid/tangzichen/wikitext2"
model_dir="/mnt/raid/tangzichen/gpt2"
# data_dir="/home/tangzhenheng/wikitext2"
# model_dir="/home/tangzhenheng/gpt2"
# PY="/mnt/raid/tangzhenheng/anaconda3/bin/python"
PY="/mnt/raid/tangzhenheng/anaconda3/envs/fusionai/bin/python"

export HF_ENDPOINT=https://hf-mirror.com

#pipe_seq_localsgd
# 127.0.0.1 localhost
# 127.0.1.1 gpu9

# 10.0.0.11 gpu1
# 10.0.0.12 gpu2
# 10.0.0.13 gpu3
# 10.0.0.14 gpu4
# 10.0.0.15 gpu5
# 10.0.0.16 gpu6
# 10.0.0.17 gpu7
# 10.0.0.18 gpu8
# 10.0.0.19 gpu9
# 10.0.0.20 gpu10
# 10.0.0.21 gpu11
# 10.0.0.22 gpu12
# 10.0.0.23 gpu13
# 10.0.0.24 gpu14
# 10.0.0.25 gpu15
# 10.0.0.26 gpu16
optimizer_name=Adam
dnn=gpt2
max_epochs=5
# add_noise=True
extra_name='gpt-convergence'

enable_wandb=false
wandb_offline=false
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name
cluster_name=GZ4090ZHTANG

# hosts=('10.0.0.20')
hosts=('localhost')
#

node_count=${#hosts[@]}
nworkers=$((4 * node_count))
nwpernode=4
nsteps_localsgd=10

# lr_decay=None
lr=0.01
lr_decay='fixed'
source train_exps/launch_transformer.sh

# lr_decay='cosine'
# node_rank=1
# lr=0.1
# source train_exps/launch_mul.sh

# lr_decay='step'
# node_rank=1
# lr=0.1
# source train_exps/launch_mul.sh
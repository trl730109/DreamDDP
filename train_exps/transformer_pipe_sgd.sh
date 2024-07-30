lr=0.00001
batch_size=4
alg='transformer_pipe_sgd'
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
dnn=llama2-124M

max_epochs=10
# add_noise=True
extra_name='llama2-124M'

enable_wandb=true
wandb_offline=true
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name
# cluster_name=GZ4090ZHTANG
cluster_name=A6000


hosts=('ibgpu4' 'ibgpu5' 'ibgpu3' 'ibgpu1')
ports=(30737 30958 31709 30715)
#
master_port=2229
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
source train_exps/launch_transformer_A6000.sh






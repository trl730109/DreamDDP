lr=0.00001
batch_size=1
alg='transformer_localsgd'
dataset='wikitext2'

data_dir="/workspace/wikitext2"
model_dir="/workspace/models/${dnn}"
PY="/workspace/pretrain/miniconda3/envs/ddp/bin/python3"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"

optimizer_name=Adam
# dnn=llama2-124M
dnn=gpt2
# Qwen2.5-7B
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
# cluster_name=GZ4090ZHTANG
cluster_name=A6000


hosts=('10.244.5.206')
ports=(22)

profile=True

master_port=2228
node_count=${#hosts[@]}
echo "$node_count"
nwpernode=8
nworkers=$((nwpernode * node_count))
nsteps_localsgd=10
ngpu_per_node=$nwpernode
# lr_decay=None

adam_beta1=0.9
# lr=0.00001
lr=0.0001

weight_decay=0.0001

lr_decay='fixed'
# source train_exps/launch_transformer_shenzhen.sh

node_rank=1
nsteps_localsgd=10

dnn=gpt2
source train_exps/launch_transformer_A6000.sh


dnn=llama2-124M
source train_exps/launch_transformer_A6000.sh




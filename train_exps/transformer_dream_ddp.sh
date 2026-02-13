lr=0.00001
batch_size=4
alg='transformer_dream_ddp'
dataset='wikitext2'

data_dir="/mnt/raid/tangzichen/wikitext2"

PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"

optimizer_name=Adam
# dnn=llama2-124M


dnn=llama2-124M

max_epochs=1
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

profile=True


hosts=('10.244.5.206')
ports=(22)
#
master_port=2456
node_count=${#hosts[@]}
echo "$node_count"
nwpernode=8
nworkers=$((nwpernode * node_count))
nsteps_localsgd=10
ngpu_per_node=$nwpernode

adam_beta1=0.9
lr=0.0001

weight_decay=0.0001

lr_decay='fixed'
enlarge=false
node_rank=1

source train_exps/launch_transformer_A6000.sh







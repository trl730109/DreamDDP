PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"

lr=0.1
batch_size=128
alg='localsgd'
script="dist_trainer.py" 

interface=eth0
# interface=ens5f0, eno0
optimizer_name=SGD
dnn=resnet50
dataset=cifar100
max_epochs=3
# add_noise=True


enable_wandb=False
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name
cluster_name=A6000
sync_momentum=false

hosts=('10.244.5.205 ')
ports=(22)
master_port=3333
#
node_count=${#hosts[@]}
nworkers=$((8 * node_count))
profile=True


nsteps_localsgd=10
ngpu_per_node=$nwpernode
extra_name="${node_count}Nodes"

# density=0.01
# compressor=topk

node_rank=1
nsteps_localsgd=10
# optimizer_name=Adam
lr_decay='exp'
nstepsupdate=1
lr=0.1
alg='localsgd'
sync_momentum=true
source train_exps/launch_mul.sh

# node_rank=1
# optimizer_name=SGD
# lr_decay='exp'
# lr=0.1
# alg='sgd'
# source train_exps/launch_mul.sh


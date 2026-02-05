PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"
# PY="/home/tangzhenheng/anaconda3/bin/python"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"


lr=0.1
batch_size=128
alg='pipe_sgd'

optimizer_name=SGD
dnn=resnet50
max_epochs=121
dataset=cifar100
# add_noise=True
extra_name="Mea_${node_count}Nodes"
interface=eth0
enable_wandb=False
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
cluster_name=A6000

hosts=('10.244.4.101')
ports=(22)

node_count=${#hosts[@]}
nworkers=$((4 * node_count))

#nsteps_localsgd=20
# lr_decay='linear'
# lr_decay=None

# lr=0.1
# node_rank=1
# lr_decay='exp'
# source train_exps/launch_mul.sh

scalar=2
lr=0.1
lr=$(echo "$lr * sqrt($scalar)" | bc -l)
node_rank=1
lr_decay='exp'
source train_exps/launch_mul.sh

# lr=0.3
# node_rank=1
# lr_decay='exp'
# source train_exps/launch_mul.sh

# lr=1
# node_rank=1
# lr_decay='exp'
# source train_exps/launch_mul.sh

# interface=ens5f0
# node_rank=1
# source train_exps/launch_mul.sh
# lr=0.1
# lr_decay='cosine'
# source train_exps/launch_mul.sh

# lr_decay='step'
# node_rank=1
# lr=0.1
# source train_exps/launch_mul.sh
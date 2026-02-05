PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"


lr=0.1
batch_size=128
alg='pipe_seq_localsgd'

optimizer_name=SGD
dnn=resnet50
max_epochs=1
dataset=cifar100
# add_noise=True
extra_name='test'
interface=eth0
enable_wandb=False
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac

exp_name=$exp_name
cluster_name=A6000

#hosts=('10.0.0.20')
hosts=('10.244.4.101')
ports=(22)
master_port=4444

node_count=${#hosts[@]}
nworkers=$((4 * node_count))

nsteps_localsgd=20

lr_decay='exp'

source train_exps/launch_mul.sh




# lr_decay='step'
# node_rank=1
# lr=0.1
# source train_exps/launch_mul.sh
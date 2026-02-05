PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"

lr=0.1
batch_size=128
alg='full_pipe_seq'

optimizer_name=SGD
dnn=resnet50
max_epochs=1
dataset=cifar100
# add_noise=True
extra_name='707'
interface=eth0


enable_wandb=False
enable_wandb
wandb_offline=False
wandb_entity=hpml-hkbu
# wandb_key=None
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
exp_name=$exp_name
cluster_name=A6000
nsteps_param_diversity=5

#hosts=('10.0.0.22')
hosts=('10.244.4.101')
ports=(22)
master_port=5555

node_count=${#hosts[@]}
nworkers=$((4 * node_count))

alg='full_pipe_seq'
node_rank=1
lr_decay='exp'
group_num=10
lr=0.1
source train_exps/launch_mul.sh
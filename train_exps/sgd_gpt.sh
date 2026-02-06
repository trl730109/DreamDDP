PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0"

lr=0.00001
batch_size=1
alg='sgd'
dataset='wikitext2'
script='dist_trainer.py'
data_dir="/data2/share/wikitext2"
model_dir="/data2/share/tzc_gpt2/gpt2"
PY="/mnt/sdb/tangzhenheng/miniconda3/envs/DDP_Train/bin/python3"
pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"

optimizer_name=Adam
# dnn=llama2-124M
dnn=gpt2

max_epochs=10


enable_wandb=true
wandb_offline=false
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name
cluster_name=A6000
sync_momentum=false


hosts=('10.244.4.101')
ports=(22)

master_port=2228
node_count=${#hosts[@]}
extra_name="${node_count}Nodes"
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
source train_exps/launch_mul.sh


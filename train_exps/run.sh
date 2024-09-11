lr=0.00001
batch_size=1
alg='localsgd'
dataset='wikitext2'
script='dist_trainer_new.py'
data_dir="/data2/share/wikitext2"
model_dir="/data2/share/tzc_gpt2/gpt2"
PY="/mnt/sdb/tangzhenheng/miniconda3/envs/DDP_Train/bin/python3"
pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"

# wandb related
enable_wandb=true
wandb_offline=false
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
cluster_name=GZ_A6000

optimizer_name=Adam
# dnn=llama2-124M
dnn=gpt2
sync_momentum=false
check_param_diversity=false
nsteps_param_diversity=5

max_epochs=10
nsteps_localsgd=10


hosts=('10.120.17.54')
master_port=2228
node_count=${#hosts[@]}
extra_name="${node_count}Nodes"
echo "$node_count"
nwpernode=8
nworkers=$((nwpernode * node_count))
ngpu_per_node=$nwpernode
# lr_decay=None

# compression
density=0.01
compressor=topk

# optimizer settings
adam_beta1=0.9
lr=0.0001
weight_decay=0.0001
lr_decay='fixed'

node_rank=1
sync_momentum=true
source train_exps/launch_mul.sh


master_port=23456
dataset=openwebtext
alg=sgd
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=Adam
dnn=gpt2
lr=8e-5
batch_size=8

max_epochs=1

add_noise=false

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"
PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

enable_wandb=true
wandb_offline=true
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac

cluster_name=A6000
hosts=('ibgpu1')
ports=(30847)

node_count=${#ports[@]}
nworkers=$((8 * node_count))
nwpernode=8
ngpu_per_node=$nwpernode
extra_name="${node_count}Nodes"

nstepsupdate=1
adam_beta1=0.9
adam_beta2=0.95
# lr=0.0001
weight_decay=0.0001

lr_decay='fixed'


# source fault_exps/launch.sh
param_sync_async_op=False

check_param_diversity=False
nsteps_param_diversity=5
nsteps_param_sync=20


source fault_exps/launch_A6000.sh

# gaussian_std=0.001
# extra_name="nstd$gaussian_std"
# source fault_exps/launch_A6000.sh

# gaussian_std=0.0001
# extra_name="nstd$gaussian_std"
# source fault_exps/launch_A6000.sh
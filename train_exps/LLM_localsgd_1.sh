lr=0.1
batch_size=8
# alg='localsgd'
script="dist_trainer_new.py" 

interface=ens5f0
optimizer_name=SGD
# dnn=llama2-124M
dnn=gpt2
# dataset=wikitext2
dataset=openwebtext
max_epochs=1

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"
PY="/workspace/pretrain/miniconda3/envs/pretrain/bin/python"

enable_wandb=true
wandb_offline=true
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac

cluster_name=A6000
hosts=('ibgpu3')
ports=(31949)

node_count=${#ports[@]}
nworkers=$((8 * node_count))
nwpernode=8
ngpu_per_node=$nwpernode
extra_name="${node_count}Nodes"

# density=0.01
# compressor=topk

nsteps_localsgd=10
optimizer_name=Adam

nstepsupdate=4
adam_beta1=0.9
adam_beta2=0.95
# lr=0.0001
weight_decay=0.0001

lr_decay='fixed'

alg='sgd'
sync_momentum=false
# source train_exps/launch_llm_SZ6000.sh

lr_values=(8e-5 4e-5 1e-5)
# lr_values=(4e-4)
#  2e-4 1e-4 8e-5 4e-5
for lr in "${lr_values[@]}"
do
    node_rank=1
    source train_exps/launch_llm_SZ6000.sh
done

# node_rank=1
# optimizer_name=SGD
# lr_decay='exp'
# lr=0.1
# alg='sgd'
# source train_exps/launch_mul.sh


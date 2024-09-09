lr=0.1
batch_size=128
alg='localsgd'
script="dist_trainer_new.py" 
#pipe_seq_localsgd
# 127.0.0.1 localhost
# 127.0.1.1 gpu9

# 10.0.0.11 gpu1
# 10.0.0.12 gpu2
# 10.0.0.13 gpu3
# 10.0.0.14 gpu4
# 10.0.0.15 gpu5
# 10.0.0.16 gpu6
# 10.0.0.17 gpu7
# 10.0.0.18 gpu8
# 10.0.0.19 gpu9
# 10.0.0.20 gpu10
# 10.0.0.21 gpu11
# 10.0.0.22 gpu12
# 10.0.0.23 gpu13
# 10.0.0.24 gpu14
# 10.0.0.25 gpu15
# 10.0.0.26 gpu16
interface=ens5f0
# interface=ens5f0, eno0
optimizer_name=Adam
dnn=resnet18
dataset=cifar10
max_epochs=120
# add_noise=True
extra_name="${node_count}Nodes-"

enable_wandb=true
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name
cluster_name=shenzhen
sync_momentum=true
if [ "$sync_momentum" = true ]; then
    extra_name="${extra_name}syncOpt"
fi


# hosts=('10.0.0.19' '10.0.0.16' '10.0.0.20' '10.0.0.21' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25')
# hosts=('10.0.0.11' '10.0.0.20' '10.0.0.21' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25' '10.0.0.26')
# hosts=('10.0.0.19' '10.0.0.18' '10.0.0.17' '10.0.0.20')
hosts=('10.0.0.19' '10.0.0.20' '10.0.0.21' '10.0.0.22')
#

node_count=${#hosts[@]}
nworkers=$((4 * node_count))
nsteps_localsgd=10
ngpu_per_node=$nwpernode

adam_beta1=0.9
# lr_decay='general'
# lr=0.1
# lr=$(echo "$lr * sqrt($scalar)" | bc -l)

node_rank=1
nsteps_localsgd=10
lr_decay='exp'
lr=0.1
alg='localsgd'
source train_exps/launch_mul.sh



# dnn=resnet18
# dataset=cifar10
# node_rank=1
# nsteps_localsgd=5
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh

# source train_exps/launch_mul.sh
# node_rank=1
# nsteps_localsgd=20
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh

# source train_exps/launch_mul.sh
# node_rank=1
# nsteps_localsgd=40
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh

# source train_exps/launch_mul.sh
# node_rank=1
# nsteps_localsgd=80
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh

# source train_exps/launch_mul.sh
# node_rank=1
# nsteps_localsgd=5
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh
# node_rank=1
# lr_decay='exp'
# lr=0.1
# scalar=2
# lr=$(echo "$lr * sqrt($scalar)" | bc -l)
# source train_exps/launch_mul.sh

# node_rank=1
# lr_decay='exp'
# lr=0.2
# source train_exps/launch_mul.sh

# node_rank=1
# lr_decay='exp'
# lr=0.2
# scalar=2
# lr=$(echo "$lr * sqrt($scalar)" | bc -l)
# source train_exps/launch_mul.sh

# node_rank=1
# lr_decay='exp'
# lr=0.2
# scalar=2
# lr=$(echo "$lr * sqrt($scalar)" | bc -l)
# source train_exps/launch_mul.sh

# node_rank=1
# dnn=resnet50
# dataset=cifar100
# lr_decay='exp'
# lr=0.2
# source train_exps/launch_mul.sh

# node_rank=1
# lr_decay='general'
# lr=0.2
# lr=$(echo "$lr * sqrt($scalar)" | bc -l)

# source train_exps/launch_mul.sh

# node_rank=1
# lr_decay='general'
# lr=0.4

# source train_exps/launch_mul.sh
# lr=0.2
# source train_exps/launch_mul.sh
# interface=ens5f0
# node_rank=1
# source train_exps/launch_mul.sh
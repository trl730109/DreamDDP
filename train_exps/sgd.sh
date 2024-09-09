lr=0.1
batch_size=128
alg='sgd'
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
optimizer_name=SGD
dnn=resnet18
dataset=cifar10
max_epochs=1
# add_noise=True
extra_name="${node_count}Nodes_"

enable_wandb=true
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac

exp_name=$exp_name
cluster_name=shenzhen
check_param_diversity=false
nsteps_param_diversity=5
hosts=('10.0.0.19' '10.0.0.20' '10.0.0.21' '10.0.0.22')
sync_momentum=true
if [ "$sync_momentum" = true ]; then
    extra_name="${extra_name}syncOpt"
fi
# hosts=('10.0.0.19' '10.0.0.18' '10.0.0.17' '10.0.0.20')
# hosts=('10.0.0.19' '10.0.0.18' '10.0.0.20' '10.0.0.21' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.26')
# #

node_count=${#hosts[@]}
nworkers=$((4 * node_count))
interface=eno0
# interface=ens5f0
#nsteps_localsgd=20

# lr=0.1
# lr_decay='general'
# source train_exps/launch_mul.sh

lr=0.1
interface=eno0
node_rank=1
lr_decay='exp'
source train_exps/launch_mul.sh

dnn=resnet50
dataset=cifar100
lr=0.1
interface=eno0
node_rank=1
lr_decay='exp'
source train_exps/launch_mul.sh

lr=0.1
batch_size=128
alg='pipe_seq_localsgd_warmup'
#pipe_seq_localsgd

optimizer_name=SGD
dnn=resnet50
max_epochs=121
dataset=cifar100
# add_noise=True
# extra_name='test'
interface=eno0
enable_wandb=true
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name
cluster_name=shenzhen

#hosts=('10.0.0.20')
hosts=('10.0.0.19' '10.0.0.20' '10.0.0.21' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25' '10.0.0.26')
#

node_count=${#hosts[@]}
extra_name="${node_count}Nodes"
nworkers=$((4 * node_count))

nsteps_localsgd=20
interface=eno0
node_rank=1
lr_decay='exp'
lr=0.1
source train_exps/launch_mul.sh

nsteps_localsgd=10
interface=eno0
node_rank=1
lr_decay='exp'
lr=0.01
source train_exps/launch_mul.sh

interface=eno0
alg='pipe_seq_localsgd'
node_rank=1
nsteps_localsgd=20
lr_decay='exp'
lr=0.1
source train_exps/launch_mul.sh

interface=eno0
alg='pipe_seq_localsgd'
node_rank=1
nsteps_localsgd=10
lr_decay='exp'
lr=0.1
source train_exps/launch_mul.sh

# # Resnet50
# dnn=resnet50
# dataset=cifar100
# node_rank=1
# nsteps_localsgd=20
# alg='localsgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# dnn=resnet50
# dataset=cifar100
# node_rank=1
# nsteps_localsgd=10
# alg='localsgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# dnn=resnet50
# dataset=cifar100
# node_rank=1
# nsteps_localsgd=5
# alg='localsgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# #pipe_sgd
# dnn=resnet50
# dataset=cifar100
# node_rank=1
# alg='pipe_sgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# #sgd
# dnn=resnet50
# dataset=cifar100
# node_rank=1
# alg='sgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

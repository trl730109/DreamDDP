lr=0.1
batch_size=128

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
dnn=resnet50
max_epochs=181
dataset=cifar100
# add_noise=True
extra_name='8Nodes-reverse'
interface=eno0
# interface=ens5f0
enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac

exp_name=$exp_name
cluster_name=shenzhen

#hosts=('10.0.0.22')
#'10.0.0.16' '10.0.0.20' '10.0.0.21' '10.0.0.22' 
hosts=('10.0.0.19' '10.0.0.23' '10.0.0.24' '10.0.0.25')
# hosts=('10.0.0.19' '10.0.0.11')
#

node_count=${#hosts[@]}
nworkers=$((4 * node_count))

nsteps_localsgd=20


# alg='localsgd'
# node_rank=1
# lr_decay='general'
# lr=0.1
# nsteps_localsgd=5
# interface=eno0
# source train_exps/launch_mul.sh

# alg='localsgd'
# node_rank=1
# lr_decay='general'
# lr=0.1
# nsteps_localsgd=10
# interface=eno0
# source train_exps/launch_mul.sh

# alg='localsgd'
# node_rank=1
# lr_decay='general'
# lr=0.1
# nsteps_localsgd=20
# interface=eno0
# source train_exps/launch_mul.sh


# node_rank=1
# alg='pipe_sgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# node_rank=1
# alg='pipe_sgd'
# lr_decay='general'
# lr=0.1
# interface=ens5f0
# source train_exps/launch_mul.sh

#sgd
# node_rank=1
# alg='sgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# node_rank=1
# alg='sgd'
# lr_decay='general'
# lr=0.1
# interface=ens5f0
# source train_exps/launch_mul.sh

# #pipe_Seq_localsgd
# dnn=resnet18
# dataset=cifar10
# node_rank=1
# nsteps_localsgd=5
# alg='pipe_seq_localsgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# dnn=resnet18
# dataset=cifar10
# node_rank=1
# nsteps_localsgd=10
# alg='pipe_seq_localsgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# dnn=resnet18
# dataset=cifar10
# node_rank=1
# nsteps_localsgd=15
# alg='pipe_seq_localsgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# dnn=resnet18
# dataset=cifar10
# node_rank=1
# nsteps_localsgd=15
# alg='pipe_seq_localsgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# node_rank=1
# nsteps_localsgd=40
# alg='pipe_seq_localsgd'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh


# group_num=6
# node_rank=1
# alg='full_pipe_seq'
# lr_decay='general'
# lr=0.1
# interface=eno0
# source train_exps/launch_mul.sh

# node_rank=1
# group_num=3
# alg='full_pipe_seq'
# lr_decay='general'
# lr=0.1
# interface=ens5f0
# source train_exps/launch_mul.sh
# #ens5f0


# Resnet50
dnn=resnet50
dataset=cifar100
node_rank=1
alg='localsgd'
lr_decay='general'
lr=0.1
interface=eno0
source train_exps/launch_mul.sh

# dnn=resnet50
# alg='localsgd'
# node_rank=1
# lr_decay='general'
# lr=0.1
# interface=ens5f0
# source train_exps/launch_mul.sh

#pipe_sgd
dnn=resnet50
dataset=cifar100
node_rank=1
alg='pipe_sgd'
lr_decay='general'
lr=0.1
interface=eno0
source train_exps/launch_mul.sh

# dnn=resnet50
# node_rank=1
# alg='pipe_sgd'
# lr_decay='general'
# lr=0.1
# interface=ens5f0
# source train_exps/launch_mul.sh

#sgd
dnn=resnet50
dataset=cifar100
node_rank=1
alg='sgd'
lr_decay='general'
lr=0.1
interface=eno0
source train_exps/launch_mul.sh

# dnn=resnet50
# node_rank=1
# alg='sgd'
# lr_decay='general'
# lr=0.1
# interface=ens5f0
# source train_exps/launch_mul.sh

#pipe_Seq_localsgd
dnn=resnet50
dataset=cifar100
nsteps_localsgd=20
node_rank=1
alg='pipe_seq_localsgd'
lr_decay='general'
lr=0.1
interface=eno0
source train_exps/launch_mul.sh

dnn=resnet50
dataset=cifar100
nsteps_localsgd=10
node_rank=1
alg='pipe_seq_localsgd'
lr_decay='general'
lr=0.1
interface=eno0
source train_exps/launch_mul.sh

dnn=resnet50
dataset=cifar100
nsteps_localsgd=5
node_rank=1
alg='pipe_seq_localsgd'
lr_decay='general'
lr=0.1
interface=eno0
source train_exps/launch_mul.sh

# dnn=resnet50
# dataset=cifar100
# node_rank=1
# alg='pipe_seq_localsgd'
# lr_decay='general'
# lr=0.1
# interface=ens5f0
# source train_exps/launch_mul.sh
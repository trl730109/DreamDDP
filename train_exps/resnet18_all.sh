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
dnn=resnet18
max_epochs=121
dataset=cifar10
# add_noise=True
# extra_name='test'
interface=eno0
enable_wandb=true
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
interface=eno0
exp_name=$exp_name
cluster_name=shenzhen
check_param_diversity=false
nsteps_param_diversity=5
#hosts=('10.0.0.20')
hosts=('10.0.0.19' '10.0.0.11' '10.0.0.12' '10.0.0.20' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25' )
#
node_count=${#hosts[@]}
# extra_name="Diversity_${node_count}Nodes"
extra_name="${node_count}Nodes"
nworkers=$((4 * node_count))

# alg='pipe_seq_localsgd'
# nsteps_localsgd=20
# node_rank=1
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh


# alg='localsgd'
# nsteps_localsgd=20
# node_rank=1
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh

# alg='sgd'
# node_rank=1
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh

# alg='pipe_sgd'
# node_rank=1
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh

# alg='pipe_sgd'
# node_rank=1
# lr_decay='exp'
# lr=0.1
# source train_exps/launch_mul.sh

# alg='full_pipe_seq'
# node_rank=1
# lr_decay='exp'
# group_num=5
# lr=0.1
# source train_exps/launch_mul.sh

alg='dream_ddp'
node_rank=1
lr_decay='exp'
group_num=5
lr=0.1
source train_exps/launch_mul.sh

alg='dream_ddp'
node_rank=1
lr_decay='exp'
group_num=10
lr=0.1
source train_exps/launch_mul.sh

alg='dream_ddp'
node_rank=1
lr_decay='exp'
group_num=12
lr=0.1
source train_exps/launch_mul.sh

alg='dream_ddp'
node_rank=1
lr_decay='exp'
group_num=15
lr=0.1
source train_exps/launch_mul.sh



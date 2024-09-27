lr=0.1
batch_size=128
alg='pipe_sgd'
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
max_epochs=121
dataset=cifar100
# add_noise=True
extra_name="Mea_${node_count}Nodes"
interface=eno0
enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
cluster_name=shenzhen

#hosts=('10.0.0.20')
# hosts=('10.0.0.18' '10.0.0.26')
hosts=('10.0.0.19' '10.0.0.11' '10.0.0.12' '10.0.0.20' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25' )
#

node_count=${#hosts[@]}
nworkers=$((4 * node_count))

#nsteps_localsgd=20
# lr_decay='linear'
# lr_decay=None

# lr=0.1
# node_rank=1
# lr_decay='exp'
# source train_exps/launch_mul.sh

scalar=2
lr=0.1
lr=$(echo "$lr * sqrt($scalar)" | bc -l)
node_rank=1
lr_decay='exp'
source train_exps/launch_mul.sh

# lr=0.3
# node_rank=1
# lr_decay='exp'
# source train_exps/launch_mul.sh

# lr=1
# node_rank=1
# lr_decay='exp'
# source train_exps/launch_mul.sh

# interface=ens5f0
# node_rank=1
# source train_exps/launch_mul.sh
# lr=0.1
# lr_decay='cosine'
# source train_exps/launch_mul.sh

# lr_decay='step'
# node_rank=1
# lr=0.1
# source train_exps/launch_mul.sh
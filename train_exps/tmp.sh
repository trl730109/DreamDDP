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
max_epochs=181
# add_noise=True
extra_name='707'
script=test_nccl.py
enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac

exp_name=$exp_name
cluster_name=shenzhen

#hosts=('10.0.0.22')
hosts=('10.0.0.19' '10.0.0.23' '10.0.0.24' '10.0.0.25')
#

node_count=${#hosts[@]}
nworkers=$((4 * node_count))
interface=eno0
# interface=ens5f0
#nsteps_localsgd=20

lr=0.1
lr_decay='general'
source train_exps/launch_mul.sh
# lr_decay=None
# lr=0.1
# lr_decay='linear'
# source train_exps/launch_mul.sh

# lr=0.1
# lr_decay='cosine'
# source train_exps/launch_mul.sh

# alg='pipe'
# # lr_decay='cosine'
# node_rank=1
# # lr=0.1
# source train_exps/launch_mul.sh

# lr_decay='step'
# node_rank=1
# lr=0.1
# source train_exps/launch_mul.sh
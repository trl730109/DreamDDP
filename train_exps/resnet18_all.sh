lr=0.1
batch_size=128
alg='localsgd'
script="dist_trainer_new.py" 
#pipe_seq_localsgd

interface=ens5f0
# interface=ens5f0, eno0
optimizer_name=SGD
dnn=resnet18
dataset=cifar10
max_epochs=120
# add_noise=True


enable_wandb=true
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
check_param_diversity=false
nsteps_param_diversity=5
exp_name=$exp_name
cluster_name=shenzhen
sync_momentum=false

# hosts=('10.0.0.19' '10.0.0.16' '10.0.0.20' '10.0.0.21' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25')
# hosts=('10.0.0.11' '10.0.0.20' '10.0.0.21' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25' '10.0.0.26')
# hosts=('10.0.0.19' '10.0.0.18' '10.0.0.17' '10.0.0.20')
hosts=('10.0.0.19' '10.0.0.20' '10.0.0.21' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25' '10.0.0.26')
#
node_count=${#hosts[@]}
nworkers=$((4 * node_count))
nsteps_localsgd=10
ngpu_per_node=$nwpernode
extra_name="${node_count}Nodes"

# density=0.01
# compressor=topk

node_rank=1
nsteps_localsgd=10
optimizer_name=Adam
lr_decay='exp'
nstepsupdate=1
lr=0.001
# global_lr=0.1
alg='train_with_global_momentum'
sync_momentum=false
# source train_exps/launch_mul.sh


# global_lrs=(0.1 0.01)
global_lrs=(0.1 0.01 0.003 0.001)
for glr in "${global_lrs[@]}"
do
    alg='train_with_global_momentum'
    node_rank=1
    global_lr=$glr 
    echo "Training with global learning rate: $glr"
    source train_exps/launch_mul.sh
done

# # global_lrs=(0.1 0.01)
# lrs=(0.003 0.001)
# for llr in "${lrs[@]}"
# do
#     echo "Training with global learning rate: $llr"
#     alg='localsgd'
#     optimizer_name=Adam
#     sync_momentum=false
#     node_rank=1
#     lr=$llr
#     source train_exps/launch_mul.sh
# done



# node_rank=1
# optimizer_name=Adam
# lr_decay='exp'
# lr=0.1
# alg='sgd'
# source train_exps/launch_mul.sh
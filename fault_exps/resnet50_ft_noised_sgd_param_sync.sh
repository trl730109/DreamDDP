

master_port=12345

alg=sgd_with_sync
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=SGD
dnn=resnet18
dataset=cifar10
data_dir=~/datasets/cifar10
lr=0.1
batch_size=128

max_epochs=111

add_noise=True

# enable_wandb=False
enable_wandb=True
wandb_offline=True
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name
# cluster_name=gpuhome
# cluster_name=scigpu
# hosts=('scigpu14')
cluster_name=shenzhen
# hosts=('localhost')

# cluster_name=esetstore
# hosts=('gpu3')
param_sync_async_op="${param_sync_async_op:-False}"
nsteps_param_diversity=5

ngpu_per_node=4
nwpernode=4

hosts=('10.0.0.11' '10.0.0.12' '10.0.0.13' '10.0.0.15' '10.0.0.16' '10.0.0.18' '10.0.0.20' '10.0.0.21')


values=(5 10 50 100)

for nsteps_param_sync in "${values[@]}"
do
    gaussian_std=0.0001
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh

    gaussian_std=0.001
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh

    gaussian_std=0.01
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh

    gaussian_std=0.1
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh

    gaussian_std=1.0
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh
done

# gaussian_std=0.0001
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh

# gaussian_std=0.001
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh

# gaussian_std=0.01
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh

# gaussian_std=0.1
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh

# gaussian_std=1.0
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh

values=(2 5 8)

# hosts=('10.0.0.11' '10.0.0.12' '10.0.0.13' '10.0.0.15' '10.0.0.16' '10.0.0.18' '10.0.0.20' '10.0.0.21')
# param_sync_async_op=False

# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.01
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh
# done

# param_sync_async_op=True

# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.01
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done

# hosts=('10.0.0.11' '10.0.0.12' '10.0.0.13' '10.0.0.15')
# param_sync_async_op=False

# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.01
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done

# param_sync_async_op=True
# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.01
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done

# hosts=('10.0.0.11' '10.0.0.12')
# param_sync_async_op=False

# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.01
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done

# param_sync_async_op=True
# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.01
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done

hosts=('10.0.0.11')
# param_sync_async_op=False

# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.01
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done

param_sync_async_op=True
for nsteps_param_sync in "${values[@]}"
do
    gaussian_std=0.01
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh

done
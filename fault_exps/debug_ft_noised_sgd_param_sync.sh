

master_port=12345

alg=sgd_with_sync
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=SGD
dnn=resnet18
lr=0.1
batch_size=32

max_epochs=111

add_noise=True

# enable_wandb=False
enable_wandb=False
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name
# cluster_name=gpuhome
# cluster_name=scigpu
# hosts=('scigpu14')
cluster_name=GZ4090
hosts=('localhost')

# cluster_name=esetstore
# hosts=('gpu3')
param_sync_async_op=False
param_sync=detect_base

nsteps_param_diversity=5
nsteps_param_sync=20

pre_cmd="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com"


values=(5)
# values=(5 10 50 100)
# values=(10 50)
# nsteps_param_sync=100

for nsteps_param_sync in "${values[@]}"
do
    # gaussian_std=0.0001
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

    # gaussian_std=0.001
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

    gaussian_std=0.01
    extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    source fault_exps/launch.sh

    # gaussian_std=0.1
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh

    # gaussian_std=1.0
    # extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
    # source fault_exps/launch.sh
done
# gaussian_std=10.0
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh


# gaussian_std=100.0
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh


# gaussian_std=1000.0
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh









# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  






























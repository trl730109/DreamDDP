PY=/workspace/pretrain/miniconda3/envs/ddp/bin/python

master_port=12999
alg=sgd
gaussian_mu=0.0
gaussian_std=0.0001
optimizer_name=SGD
dnn=resnet18
lr=0.1
batch_size=128

nworkers=8
ngpu_per_node=8
max_epochs=111



enable_wandb=True
wandb_offline=True
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name
# cluster_name=gpuhome
# cluster_name=scigpu
# hosts=('localhost')
# cluster_name=GZ4090
# hosts=('localhost')
cluster_name=A6000
hosts=('ibgpu3')
ports=(32193)

# cluster_name=esetstore
# hosts=('gpu6')



# source fault_exps/launch.sh
param_sync_async_op=False

# nsteps_param_diversity=5
# nsteps_param_sync=20
# noise_type=burst
# burst_freq=500
# burst_magnitude=1.0
# source fault_exps/launch.sh


dnn=resnet18
dataset=cifar10
nsteps_param_diversity=5
nsteps_param_sync=20
add_noise=True
noise_type=fix
gaussian_std=0.0001
gaussian_mu=0.0 
# burst_freq=500
# burst_magnitude=10
flip_prob=0.1
bit_flipping=True
params_flipping_rate=0.1
bit_flipping_interval=500   
extra_name="params_flip${params_flipping_rate}-flip_prob${flip_prob}-bit_flip_interval${bit_flipping_interval}-nstd${gaussian_std}"
source fault_exps/launch_A6000.sh

# burst_magnitude=100
# source fault_exps/launch_A6000.sh


# gaussian_std=0.0001
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh

# gaussian_std=0.001
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh

# gaussian_std=0.01
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh


# gaussian_std=0.1
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh


# gaussian_std=1.0
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh


# alg=sgd_with_sync
# gaussian_mu=0.0
# gaussian_std=0.001
# optimizer_name=SGD
# dnn=resnet18
# lr=0.1
# batch_size=128

# max_epochs=111

# add_noise=True
# param_sync=detect_base

# gaussian_std=0.0001
# extra_name="nstd$gaussian_std-SyncPdetect"
# source fault_exps/launch.sh

# gaussian_std=0.001
# extra_name="nstd$gaussian_std-SyncPdetect"
# source fault_exps/launch.sh

# gaussian_std=0.01
# extra_name="nstd$gaussian_std-SyncPdetect"
# source fault_exps/launch.sh

# gaussian_std=0.1
# extra_name="nstd$gaussian_std-SyncPdetect"
# source fault_exps/launch.sh

# gaussian_std=1.0
# extra_name="nstd$gaussian_std-SyncPdetect"
# source fault_exps/launch.sh


# gaussian_std=10.0
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh


# gaussian_std=100.0
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh


# gaussian_std=1000.0
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh









# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  
































master_port=23456

alg=sgd
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=SGD
dnn=resnet18
lr=0.1
batch_size=128

max_epochs=111

add_noise=True

enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name
# cluster_name=gpuhome
# cluster_name=scigpu
# hosts=('localhost')
cluster_name=scigpu
hosts=('scigpu14')


# cluster_name=esetstore
# hosts=('gpu6')



# source fault_exps/launch.sh
param_sync_async_op=False

nsteps_param_diversity=5
nsteps_param_sync=20


gaussian_std=0.0001
extra_name="nstd$gaussian_std"
source fault_exps/launch.sh

gaussian_std=0.001
extra_name="nstd$gaussian_std"
source fault_exps/launch.sh

gaussian_std=0.01
extra_name="nstd$gaussian_std"
source fault_exps/launch.sh


gaussian_std=0.1
extra_name="nstd$gaussian_std"
source fault_exps/launch.sh


gaussian_std=1.0
extra_name="nstd$gaussian_std"
source fault_exps/launch.sh


alg=sgd_with_sync
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=SGD
dnn=resnet18
lr=0.1
batch_size=128

max_epochs=111

add_noise=True
param_sync=detect_base

gaussian_std=0.0001
extra_name="nstd$gaussian_std-SyncPdetect"
source fault_exps/launch.sh

gaussian_std=0.001
extra_name="nstd$gaussian_std-SyncPdetect"
source fault_exps/launch.sh

gaussian_std=0.01
extra_name="nstd$gaussian_std-SyncPdetect"
source fault_exps/launch.sh

gaussian_std=0.1
extra_name="nstd$gaussian_std-SyncPdetect"
source fault_exps/launch.sh

gaussian_std=1.0
extra_name="nstd$gaussian_std-SyncPdetect"
source fault_exps/launch.sh


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






























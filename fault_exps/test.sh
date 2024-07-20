

master_port=11111

alg=sgd
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=SGD
dnn=resnet18
lr=0.1
batch_size=128

max_epochs=111

add_noise=True

enable_wandb=False
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name
# cluster_name=gpuhome
cluster_name=scigpu
hosts=('scigpu14')

# cluster_name=esetstore
# hosts=('gpu3')


# param_sync=detect_base
param_sync=fix

alg=sgd_with_sync

# alg=sgd

# gaussian_std=0.000001

# gaussian_std=0.001

gaussian_std=0.01

# gaussian_std=0.1


# gaussian_std=10.0
# gaussian_std=10000.0
param_sync_async_op=False

nsteps_param_diversity=1
nsteps_param_sync=20
# gaussian_std=10.0
extra_name="nstd$gaussian_std-Psync${param_sync}-S${nsteps_param_sync}-Disp${nsteps_param_diversity}"
source fault_exps/launch.sh
# source fault_exps/launch_debug.sh


# gaussian_std=100.0
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh


# gaussian_std=1000.0
# extra_name="nstd$gaussian_std"
# source fault_exps/launch.sh









# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  


































alg=sgd
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=SGD
dnn=resnet50
lr=0.1
batch_size=128
dataset=cifar100
data_dir=~/datasets/cifar100

max_epochs=3

add_noise=False

enable_wandb=True
wandb_offline=True
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name
cluster_name=shenzhen
param_sync_async_op="${param_sync_async_op:-False}"

ngpu_per_node=4
nwpernode=4

# hosts=('localhost')
source fault_exps/launch.sh



# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  

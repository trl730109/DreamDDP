



alg=sgd
gaussian_mu=0.0
gaussian_std=0.001
optimizer_name=SGD
dnn=resnet18
lr=0.1
batch_size=128

max_epochs=181

add_noise=False

enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name
cluster_name=gpuhome

hosts=('gpu23')
source fault_exps/launch.sh



lr=0.01

source fault_exps/launch.sh




# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  






























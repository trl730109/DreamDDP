lr=0.1
batch_size=128
alg='localsgd'
#pipe_seq_localsgd

optimizer_name=SGD
dnn=resnet18
lr=0.1
max_epochs=91
# add_noise=True
extra_name='localsgd-Convergence'

enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
exp_name=$exp_name
cluster_name=shenzhen

# hosts=('gpu9' 'gpu3' 'gpu11' 'gpu12')
hosts=('gpu11')
node_count=${#hosts[@]}


nsteps_localsgd=20
source train_exps/launch.sh

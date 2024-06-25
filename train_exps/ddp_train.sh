


lr=0.1
batch_size=128
alg=pipe_seq_localsgd
# alg=test
# gaussian_mu=0.0
# gaussian_std=0.001
optimizer_name=SGD
dnn=resnet18
lr=0.1
max_epochs=181
sync='sum'
# add_noise=True
extra_name='625'
# if [ "$alg" = "pipe_seq_localsgd" ]; then
#     exp_name="${extra_name}-${alg}-${sync}-${dnn}-${optimizer_name}"
# else
#     exp_name="${extra_name}-${alg}-${dnn}-${optimizer_name}"
# fi


enable_wandb=True
wandb_offline=False
wandb_entity=hpml-hkbu
wandb_key=174615c3e7f0204e9374d7ace7a3e91c580124ac
exp_name=$exp_name
cluster_name=shenzhen

hosts=('gpu3')

# sync='avg'
# source train_exps/launch.sh

sync='sum'
source train_exps/launch.sh

# sync='avg'
# source train_exps/launch.sh




# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  






























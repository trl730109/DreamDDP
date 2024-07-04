


lr=0.1
batch_size=128
alg='transformer_localsgd'
#pipe_seq_localsgd
# alg=test
# gaussian_mu=0.0
# gaussian_std=0.001
optimizer_name=Adam
dnn=gpt2
dataset=wikitext2
lr=0.1
max_epochs=91
sync='avg'
# add_noise=True
extra_name='localsgd-Convergence'
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

# hosts=('gpu9' 'gpu3' 'gpu11' 'gpu12')
hosts=('gpu11')
node_count=${#hosts[@]}
# sync='avg'
# source train_exps/launch.sh

# sync='avg'
# source train_exps/launch.sh

# sync='avg'
# source train_exps/launch.sh

nsteps_localsgd=20
source train_exps/launch.sh


# nsteps_localsgd=1600
# source train_exps/launch.sh

# nsteps_localsgd=2000
# source train_exps/launch.sh

# nsteps_localsgd=400
# source train_exps/launch.sh
# alg='sgd'
# source train_exps/launch.sh



# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  






























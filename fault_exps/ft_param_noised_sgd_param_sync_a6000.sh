PY=/workspace/pretrain/miniconda3/envs/ddp/bin/python
# PY=/home/tangzhenheng/anaconda3/envs/fusionai/bin/python


master_port=13999

alg=sgd_with_sync
gaussian_mu=0.0
gaussian_std=0.0001
optimizer_name=SGD
dnn=resnet18
dataset=cifar10
lr=0.1
batch_size=128

nworkers=8
ngpu_per_node=8
max_epochs=111

add_noise=True

# enable_wandb=False
enable_wandb=True
wandb_offline=True
wandb_entity=hpml-hkbu
wandb_key=5edd8acc594b95a0b4f58e39b2243143f03c65a0
exp_name=$exp_name

cluster_name=A6000
hosts=('ibgpu6')
ports=(30785)

param_sync_async_op=False
values=(10)

nsteps_param_diversity=5
nsteps_param_sync=10
noise_type=fix
# burst_freq=500
# burst_magnitude=1.0

add_noise_params=True
add_noise_params_mu=0.0
add_noise_params_std=0.01
avg_params_clip=False
# avg_params_clip_threshold=0.01

param_sync=fix
gaussian_std=0.1

for nsteps_param_sync in "${values[@]}"
do
    param_sync=fix
    add_noise_params_std=0.001
    extra_name="nstd$gaussian_std-param_noised${add_noise_params_std}-SyncP${nsteps_param_sync}-param_sync${param_sync}-avg_params_clip${avg_params_clip}"
    source fault_exps/launch_A6000.sh

    param_sync=fix
    add_noise_params_std=0.1
    extra_name="nstd$gaussian_std-param_noised${add_noise_params_std}-SyncP${nsteps_param_sync}-param_sync${param_sync}-avg_params_clip${avg_params_clip}"
    source fault_exps/launch_A6000.sh

    # param_sync=fix
    # add_noise_params_std=10
    # extra_name="nstd$gaussian_std-param_noised${add_noise_params_std}-SyncP${nsteps_param_sync}-param_sync${param_sync}-avg_params_clip${avg_params_clip}"
    # source fault_exps/launch_A6000.sh
done


avg_params_clip=True
# avg_params_clip_threshold=0.1
clip_values=(1 3 10)
for avg_params_clip_threshold in "${clip_values[@]}"
do  
    param_sync=fix
    add_noise_params_std=0.001
    extra_name="nstd$gaussian_std-param_noised${add_noise_params_std}-SyncP${nsteps_param_sync}-param_sync${param_sync}-avg_params_clip${avg_params_clip}-avg_params_clip_threshold${avg_params_clip_threshold}"
    source fault_exps/launch_A6000.sh

    param_sync=fix
    add_noise_params_std=0.1
    extra_name="nstd$gaussian_std-param_noised${add_noise_params_std}-SyncP${nsteps_param_sync}-param_sync${param_sync}-avg_params_clip${avg_params_clip}-avg_params_clip_threshold${avg_params_clip_threshold}"
    source fault_exps/launch_A6000.sh

    param_sync=fix
    add_noise_params_std=10
    extra_name="nstd$gaussian_std-param_noised${add_noise_params_std}-SyncP${nsteps_param_sync}-param_sync${param_sync}-avg_params_clip${avg_params_clip}-avg_params_clip_threshold${avg_params_clip_threshold}"
    source fault_exps/launch_A6000.sh
done


# alg=sgd
# source fault_exps/launch.sh

# alg=sgd_with_sync
# param_sync=fix



# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.0001
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done


# param_sync=detect_base
# nsteps_param_sync=10
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh


# burst_magnitude=0.1
# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.0001
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done

# param_sync=detect_base
# nsteps_param_sync=10
# extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
# source fault_exps/launch.sh

# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.0001
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh





# hosts=('gpu23') alg=$alg add_noise=$add_noise gaussian_std=$gaussian_std  






# dnn=resnet50


# burst_magnitude=1.0
# for nsteps_param_sync in "${values[@]}"
# do
#     gaussian_std=0.0001
#     extra_name="nstd$gaussian_std-SyncP${nsteps_param_sync}"
#     source fault_exps/launch.sh

# done


































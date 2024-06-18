#!/bin/bash
export HOROVOD_WITH_MPI=1
#export HOROVOD_WITH_GLOO=1
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-1.0}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"
nwpernode=4
nstepsupdate=1
overlap_scalar=2
strategy='average'
nsteps_localsgd=1
optimizer_name='SGD'
alg='sgd'
add_noise='false'
PY=~/miniconda3/envs/DDP/bin/python3

MPIPATH=~/miniconda3/envs/DDP/

GRADSPATH=./logs/tzc

# Define arrays of means and standard deviations
# Lower values for potential convergence
means=(0.0)
stds=(0.0001)
#stds=(0.0001)
# Higher values for potential non-convergence
# (Including a combination known to fail: mean=0.1, std=0.03)


# Loop over each mean and std
for gaussian_mu in "${means[@]}"; do
    for gaussian_std in "${stds[@]}"; do
        echo "Running with Gaussian mu=${gaussian_mu}, std=${gaussian_std}"
        horovodrun -np $nworkers -H localhost:4 $PY horovod_trainer.py \
            --alg $alg \
            --add_noise $add_noise \
            --gaussian_mu $gaussian_mu \
            --gaussian_std $gaussian_std \
            --optimizer_name $optimizer_name \
            --nsteps_localsgd $nsteps_localsgd \
            --strategy $strategy \
            --overlap_scalar $overlap_scalar \
            --dnn $dnn --dataset $dataset \
            --max-epochs $max_epochs \
            --batch-size $batch_size \
            --nworkers $nworkers \
            --data-dir $data_dir \
            --lr $lr \
            --nsteps-update $nstepsupdate \
            --nwpernode $nwpernode \
            --density $density \
            --compressor $compressor \
            --threshold $threshold \
            --saved-dir $GRADSPATH \
            --momentum-correction $momentum_correction
    done
done

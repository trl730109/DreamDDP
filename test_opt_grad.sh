#!/bin/bash
export HOROVOD_WITH_MPI=1
#export HOROVOD_WITH_GLOO=1

# Default settings
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
PY=~/miniconda3/envs/DDP/bin/python3

MPIPATH=~/miniconda3/envs/DDP/
GRADSPATH=./logs/tzc

# Optimizers to iterate over
optimizers=('Adam' 'SGD' 'AdamW')

# Loop through each optimizer
for optimizer_name in "${optimizers[@]}"
do
    echo "Running training with optimizer: $optimizer_name"
    horovodrun -np $nworkers -H localhost:4 $PY horovod_trainer.py \
        --optimizer_name $optimizer_name \
        --nsteps_localsgd $nsteps_localsgd \
        --strategy $strategy \
        --overlap_scalar $overlap_scalar \
        --dnn $dnn \
        --dataset $dataset \
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

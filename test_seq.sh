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
strategy='ties'
nsteps_localsgd=20
optimizer_name='SGD'
alg='seq'
PY=~/miniconda3/envs/DDP/bin/python3

MPIPATH=~/miniconda3/envs/DDP/

GRADSPATH=./logs/tzc

# Loop through each strategy and run the training script
horovodrun -np $nworkers -H localhost:4 $PY horovod_trainer.py \
    --alg $alg \
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
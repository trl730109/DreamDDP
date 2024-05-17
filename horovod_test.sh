#!/bin/bash
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-0.01}"
threshold="${threshold:-524288000}"
compressor="${compressor:-topk}"
momentum_correction="${momentum_correction:-0}"
nwpernode=4
nstepsupdate=1
nsteps_localsgd=10
#PY=/home/comp/csshshi/anaconda2/bin/python
#PY=/usr/local/bin/python
PY=~/miniconda3/envs/DDP/bin/python3
#PY=/home/comp/csshshi/anaconda2/bin/python
MPIPATH=~/miniconda3/envs/DDP/
#MPIPATH=/home/comp/csshshi/local/openmpi3.1.1
#MPIPATH=/home/esetstore/.local/openmpi-4.0.1
#PY=/home/esetstore/anaconda3/bin/python
#PY=python3
#PY=/home/comp/15485625/pytorch1.4/bin/python
#GRADSPATH=./logs/iclr
GRADSPATH=./logs/tzc

#cd $CURR_PATH
#HOROVOD_CYCLE_TIME=1 HOROVOD_FUSION_THRESHOLD=0 
#HOROVOD_TIMELINE=./logs/profile-timeline-${dnn}.json.log HOROVOD_TIMELINE_MARK_CYCLES=1 


$PY horovod_trainer.py --nsteps-localsgd $nsteps_localsgd --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nsteps-update $nstepsupdate --nwpernode $nwpernode --density $density --compressor $compressor --threshold $threshold --saved-dir $GRADSPATH --momentum-correction $momentum_correction


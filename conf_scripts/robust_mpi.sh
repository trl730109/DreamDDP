#!/bin/bash
dnn="${dnn:-resnet50}"
density="${density:-0.001}"
source exp_configs/$dnn.conf
compressor="${compressor:-topk}"
nworkers="${nworkers:-4}"
nwpernode=4
sigmascale=2.5
MPIPATH=/home/comp/csshshi/local/openmpi3.1.1
#MPIPATH=/home/comp/csshshi/anaconda3
PY=/home/comp/csshshi/anaconda2/bin/python
#PY=/home/comp/csshshi/anaconda3/bin/python
#MPIPATH=/home/shshi/local/openmpi3.1.1
#PY=/home/shshi/anaconda2/bin/python
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers --bind-to none -map-by slot \
    -x LD_LIBRARY_PATH \
    $PY -m mpi4py robust_trainer.py --dnn $dnn --dataset $dataset --max-epochs $max_epochs --batch-size $batch_size --nworkers $nworkers --data-dir $data_dir --lr $lr --nwpernode $nwpernode --nsteps-update $nstepsupdate --compression --sigma-scale $sigmascale --density $density --compressor $compressor --gpu 1
    #$PY -m mpi4py allreducer.py

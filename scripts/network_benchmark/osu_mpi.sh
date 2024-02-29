#!/bin/bash
size="${size:-65536}"
iter="${iter:-1000}"
nnodes="${nnodes:-2}"
#PY=/home/comp/qiangwang/anaconda2/bin/python
#MPIPATH=/home/t716/blackjack/software/openmpi-3.1.0
#MPIPATH=/usr/local/openmpi/openmpi-4.0.1
MPIPATH=/home/esetstore/.local/openmpi-4.0.1
hostfile="${hostfile:-cluster$nnodes}"
$MPIPATH/bin/mpirun --prefix $MPIPATH -np $nnodes -hostfile $hostfile -bind-to none -map-by slot \
        -mca pml ob1 -mca btl ^openib \
        -mca coll_tuned_use_dynamic_rules 1 -mca coll_tuned_allreduce_algorithm 4 \
        -mca btl_tcp_if_include 192.168.0.1/24 \
        ./osu_allreduce -m $size:$size -i $iter
        #./osu_allreduce -m 524288:134217728 -i $iter

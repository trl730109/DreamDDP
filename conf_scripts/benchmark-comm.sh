#!/bin/bash
nworkers="${nworkers:-4}"
MPIPATH=/usr/local/openmpi/openmpi-4.0.1
PY=python
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers -hostfile cluster$nworkers -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -x PYTHONPATH=./ \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include em1 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=em1 \
    $PY test/comm_test.py > logs/v100-10Gbps-horovod-nccl-large-n${nworkers}.log

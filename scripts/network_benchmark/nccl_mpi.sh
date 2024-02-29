#!/bin/bash
nworkers="${nworkers:-2}"
size="${size:-26214400}"
iter="${iter:-20}"
gpuid="${gpuid:-0}"
rdma="${rdma:-0}"
#MPIPATH=/usr/local/openmpi/openmpi-4.0.1
MPIPATH=/home/esetstore/.local/openmpi-4.0.1
ALLREDUCE_BIN=/home/esetstore/downloads/nccl-tests/build/all_reduce_perf
SOCKET_IFNAME=192.168.0.1/24

if [ "$rdma" = "0" ]; then
    net_conf="-mca pml ob1 -mca btl ^openib --mca btl_openib_allow_ib 0 -mca btl_tcp_if_include $SOCKET_IFNAME -x NCCL_DEBUG=INFO -x NCCL_IB_DISABLE=1 -x NCCL_SOCKET_IFNAME=enp136s0f0,enp137s0f0 -x CUDA_VISIBLE_DEVICES=$gpuid"
elif [ "$rdma" = "1" ]; then
    net_conf="--mca pml ob1 --mca btl openib,vader,self --mca btl_openib_allow_ib 1 -mca btl_tcp_if_include ib0 --mca btl_openib_want_fork_support 1 -x LD_LIBRARY_PATH -x NCCL_IB_DISABLE=0 -x NCCL_SOCKET_IFNAME=ib0 -x NCCL_DEBUG=INFO"
else
    net_conf="--mca pml ob1 --mca btl ^openib --mca btl_openib_allow_ib 0 -mca btl_tcp_if_include ib0 --mca btl_openib_want_fork_support 1 -x LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME=ib0 -x NCCL_DEBUG=INFO -x NCCL_IB_DISABLE=1 -x NCCL_NET_GDR_LEVEL=0 -x NCCL_NET_GDR_READ=0 -x NCCL_IB_CUDA_SUPPORT=0"
fi
$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers --hostfile cluster$nworkers \
    $net_conf \
    -x NCCL_TREE_THRESHOLD=1 \
    -x NCCL_LL_THRESHOLD=1 \
    -x NCCL_TREE_THRESHOLD=1 \
    $ALLREDUCE_BIN -b $size -e $size -i 1 -g 1 -c 0  -n $iter -z 1 -w 5
    #$ALLREDUCE_BIN -b 512 -e 4M -i 32768 -g 1

#$MPIPATH/bin/mpirun --oversubscribe --prefix $MPIPATH -np $nworkers --hostfile cluster$nworkers \
#    $net_conf \
#    $ALLREDUCE_BIN -b 1M -e 4M -i 32768 -g 1
    #$ALLREDUCE_BIN -b 512 -e 256M -f 2 -g 1
    #$ALLREDUCE_BIN -b $size -e $size -i 1 -g 1 -c 0  -n $iter -z 1 -w 5

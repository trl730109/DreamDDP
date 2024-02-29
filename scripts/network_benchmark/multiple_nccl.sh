#!/bin/bash
job_num="${job_num:-4}"
size="${size:-26214400}"
iter="${iter:-1000}"
nworkers="${nworkers:-2}"
rdma="${rdma:-0}"
logRoot=logs/infocom21-v2/nccl_rdma${rdma}_job_nw${nworkers}_n${job_num}_s${size}
mkdir -p $logRoot
#for (( i=1; i<=$job_num; i++ ))
for i in `seq 1 ${job_num}`
do
    date +"%T.%N"
    gpuid=$(expr $i - 1)
    #gpuid=$(expr $gpuid % 1)
    #echo $gpuid
    rdma=$rdma gpuid=$gpuid size=$size iter=$iter nworkers=$nworkers ./nccl_mpi.sh 1>${logRoot}/nccl_job_$i.log 2>&1 &
    #rdma=$rdma gpuid=0,1,2,3 size=$size iter=$iter nworkers=$nworkers ./nccl_mpi.sh 1>${logRoot}/nccl_job_$i.log 2>&1 &
done
wait
echo "Finish and sleep 1s..."
sleep 1

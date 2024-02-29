#!/bin/bash
job_num="${job_num:-4}"
size="${size:-104857600}"
iter="${iter:-100}"
nnodes="${nnodes:-2}"
logRoot=logs/v2_job_nnodes${nnodes}_n${job_num}_s${size}
mkdir -p $logRoot
#for (( i=1; i<=$job_num; i++ ))
for i in `seq 1 ${job_num}`
do
    date +"%T.%N"
    nnodes=$nnodes size=$size iter=$iter ./osu_mpi.sh 1>${logRoot}/job_$i.log 2>&1 &
done
wait
echo "Finish and sleep 15s..."
sleep 5

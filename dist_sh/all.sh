#!/bin/bash

# Set Python and script environment
#source ../configs/envs.conf
directory=`pwd`
script="${script:-dist_trainer.py}"  # Assuming this is the PyTorch distributed training script
params="${params:-}"
echo "launch dir: $directory"

# Model and training configurations
dnn="${dnn:-resnet20}"
source exp_configs/$dnn.conf
nworkers="${nworkers:-4}"
density="${density:-1.0}"
threshold="${threshold:-524288000}"
compressor="${compressor:-none}"
momentum_correction="${momentum_correction:-0}"

# PyTorch Distributed settings
ngpu_per_node="${ngpu_per_node:-4}"
node_count=1  # Total number of nodes to use
hosts=('gpu22')  # Hosts available for training
total_host=${#hosts[@]}
master_host=${hosts[0]}  # First host as the master host

# Training settings
nwpernode=4
nstepsupdate=1
overlap_scalar=2
strategy='average'
nsteps_localsgd=20
optimizer_name='SGD'
algorithms=('sgd' 'localsgd' 'pipe')
PY=~/miniconda3/envs/DDP/bin/python3
GRADSPATH=./logs/tzc

# Loop to launch training on each node for each algorithm
for alg in "${algorithms[@]}"; do
    for node_rank in $(seq 0 $(($node_count - 1))); do
        if [ $node_rank -ge $total_host ]; then
            echo "Required nodes are out of the range: from gpu1 to gpu${total_host}"
            exit 0
        fi
        host=${hosts[$node_rank]}
        echo "Launching training on $host for algorithm $alg"
        args="$PY -m torch.distributed.run --nproc_per_node=$ngpu_per_node --nnodes=$node_count --node_rank=$node_rank --master_addr=$master_host --master_port=12345 $script \
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
            --momentum-correction $momentum_correction"
        cmd="cd $directory; $args"
        ssh $host "$cmd" &
    done
    wait  # Wait for all nodes to finish before starting next algorithm
done
echo "All training processes have completed."

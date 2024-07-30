PY=/mnt/raid/tangzhenheng/anaconda3/envs/fusionai/bin/python

directory=`pwd`

echo "launch dir: $directory"




hosts=("localhost")
ngpu_per_node="${ngpu_per_node:-4}"

node_count="${node_count:-1}"
node_rank="${node_rank:-1}"
node_rank=$(expr $node_rank - 1)  # Adjust for zero-based indexing
master_host=${hosts[$node_rank]}


i=0
while [ $i -lt $node_count ]
do
    host=${hosts[$node_rank]}
    args="NCCL_P2P_DISABLE=1 HF_ENDPOINT=https://hf-mirror.com  $PY -m torch.distributed.run --nproc_per_node=$ngpu_per_node --nnodes=$node_count \
        --node_rank=$i --master_addr=$master_host \
        --master_port=12229 \
        allreduce_hf_model.py "
    echo "$host: $args"
    cmd="cd $directory; $args"
    if [ $(expr $i + 1) -eq $node_count ]; then
        ssh $host $cmd # return until finished or interrupted
    else
        ssh $host $cmd >> /dev/null & # return immediately
    fi
    node_rank=$(expr $node_rank + 1)
    i=$(expr $i + 1)
done









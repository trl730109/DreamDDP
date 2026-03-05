#!/bin/bash

# List of GPU node IP addresses
# declare -a gpunodes=('10.244.5.206' '10.244.19.248' '10.244.1.112' '10.244.3.174')
# declare -a gpunodes=('10.244.1.115' '10.244.3.176')
declare -a gpunodes=('10.244.19.3' '10.244.3.185')
# hosts=('10.0.0.19' '10.0.0.16' '10.0.0.20' '10.0.0.21' '10.0.0.22' '10.0.0.23' '10.0.0.24' '10.0.0.25')
# 在每台节点上执行清理脚本（杀 dist_trainer_transformer.py），不要指向本脚本否则会无限递归
command="bash /workspace/tzc/DreamDDP/train_exps/killnoise.sh"

# Loop through each node and execute the command via SSH
for node_ip in "${gpunodes[@]}"
do
    echo "Executing on GPU node at $node_ip..."
    ssh "$node_ip" "$command"
done

echo "Completed executing commands on all GPU nodes."
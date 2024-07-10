#!/bin/bash

# List of GPU node IP addresses
declare -a gpunodes=(
# "10.0.0.11" # gpu1
# "10.0.0.12" # gpu2
# "10.0.0.13" # gpu3
# "10.0.0.14" # gpu4
# "10.0.0.15" # gpu5
# "10.0.0.16" # gpu6
# "10.0.0.17" # gpu7
# "10.0.0.18" # gpu8
# "10.0.0.19" # gpu9
# "10.0.0.20" # gpu10
# "10.0.0.21" # gpu11
"10.0.0.22" # gpu12
"10.0.0.23" # gpu13
"10.0.0.24" # gpu14
"10.0.0.25" # gpu15
# "10.0.0.26" # gpu16
)

# The command to run on each GPU node
command="bash ./DDP-Train/train_exps/killnoise.sh"

# Loop through each node and execute the command via SSH
for node_ip in "${gpunodes[@]}"
do
    echo "Executing on GPU node at $node_ip..."
    ssh "$node_ip" "$command"
done

echo "Completed executing commands on all GPU nodes."
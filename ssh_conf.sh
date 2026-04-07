#!/bin/bash

# === Config ===
HOST="10.249.40.11"
# 在这里填入所有端口号，用空格分隔
# PORTS=(31972 31521)
PORTS=(31731)
USER="root"
EMAIL="ztangap@connect.ust.hk"

# 1. Generate Key if not exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key..."
    ssh-keygen -t rsa -C "$EMAIL" -f ~/.ssh/id_rsa -N ""
else
    echo "SSH key already exists."
fi

# 2. Copy public key to all nodes (so you can SSH in from local machine)
echo "Copying public key..."
for port in "${PORTS[@]}"; do
    echo "  -> $HOST:$port"
    ssh-copy-id -p "$port" -o StrictHostKeyChecking=no "$USER@$HOST"
done

# 3. Copy private key to all nodes (so nodes can SSH to each other via internal IPs)
# Internal IPs (10.244.x.x) are reachable within the cluster but still require key auth.
# Training scripts SSH from master to workers using internal IPs, so every node needs the private key.
echo "Copying private key to all nodes..."
for port in "${PORTS[@]}"; do
    echo "  -> $HOST:$port"
    scp -P "$port" -o StrictHostKeyChecking=no ~/.ssh/id_rsa "$USER@$HOST:~/.ssh/id_rsa"
    ssh -p "$port" -o StrictHostKeyChecking=no "$USER@$HOST" "chmod 600 ~/.ssh/id_rsa"
done

echo "All done."
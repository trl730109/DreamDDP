#!/bin/bash

# === Config ===
HOST="10.249.40.11"
# 在这里填入所有端口号，用空格分隔
PORTS=(31972 31521)
USER="root"
EMAIL="ztangap@connect.ust.hk"

# 1. Generate Key if not exists
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key..."
    ssh-keygen -t rsa -C "$EMAIL" -f ~/.ssh/id_rsa -N ""
else
    echo "SSH key already exists."
fi

# 2. Copy Key to all ports
echo "Starting copy process..."

for port in "${PORTS[@]}"; do
    echo "Processing port $port..."
    ssh-copy-id -p "$port" -o StrictHostKeyChecking=no "$USER@$HOST"
done

echo "All done."
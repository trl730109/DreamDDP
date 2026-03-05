#!/bin/bash

# === Config ===
IFACE="eth0"
# ==============

if [ "$EUID" -ne 0 ]; then
    echo "Error: Run as root."
    exit 1
fi

CMD=$1
VAL=$2

install_tools() {
    echo "Installing tools..."
    apt-get update -qq
    apt-get install -y iproute2 iperf3 ethtool > /dev/null
    echo "Done."
}

limit_bw() {
    if [ -z "$VAL" ]; then
        echo "Usage: $0 limit <rate> (e.g., 1gbit)"
        exit 1
    fi
    # Clear old rules
    tc qdisc del dev $IFACE root 2> /dev/null
    # Set new rule
    tc qdisc add dev $IFACE root tbf rate $VAL burst 50mbit latency 400ms
    
    if [ $? -eq 0 ]; then
        echo "Limit set to $VAL on $IFACE."
    else
        echo "Failed. Check interface name."
    fi
}

clear_bw() {
    tc qdisc del dev $IFACE root 2> /dev/null
    echo "Limits cleared."
}

show_status() {
    echo "--- Interface: $IFACE ---"
    ethtool $IFACE | grep Speed
    echo "--- Active Rules ---"
    tc qdisc show dev $IFACE
}

case "$CMD" in
    install) install_tools ;;
    limit)   limit_bw ;;
    clear)   clear_bw ;;
    show)    show_status ;;
    *)
        echo "Usage: $0 {install | limit <rate> | clear | show}"
        exit 1
        ;;
esac
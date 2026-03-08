#!/bin/bash
# Usage: bash set_mul_bandwidth.sh [bandwidth]
#   bandwidth: e.g. 10gbit, 1gbit (default 1gbit when not set from pipeline/launch)

BANDWIDTH="${1:-1gbit}"

echo "Installing tools..."
bash ./set_bandwidth.sh install

echo "Showing bandwidth before setting:"
bash ./set_bandwidth.sh show


echo "Setting bandwidth to ${BANDWIDTH}..."
bash ./set_bandwidth.sh limit "$BANDWIDTH"


echo "Showing bandwidth after setting:"
bash ./set_bandwidth.sh show

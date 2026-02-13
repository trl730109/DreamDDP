#!/bin/bash

echo "Installing tools..."
bash ./set_bandwidth.sh install

echo "Showing bandwidth before setting:"
bash ./set_bandwidth.sh show


echo "Setting bandwidth to 1GBps..."
bash ./set_bandwidth.sh limit 1gbit


echo "Showing bandwidth after setting:"
bash ./set_bandwidth.sh show

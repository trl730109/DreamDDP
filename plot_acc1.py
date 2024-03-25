#!/home/your_username/miniconda3/envs/DDP/bin/python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Function to parse accuracy and loss from a log file.
def parse_metrics(log_file):
    epochs, accuracies, losses = [], [], []
    epoch_pattern = re.compile(r'Epoch (\d+), avg train acc: ([0-9.]+), lr: [0-9.]+, avg loss: ([0-9.]+)')
    with open(log_file, 'r') as file:
        for line in file:
            match = epoch_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                accuracy = float(match.group(2))
                loss = float(match.group(3))
                epochs.append(epoch)
                accuracies.append(accuracy)
                losses.append(loss)
    return epochs, accuracies, losses


# Directories containing experiment logs.
log_directories = {
    'Baseline': '/home/comp/amelieczhou/DDP-Train/logs/03-22-09:21allreduce-gwarmup-dc1-model-debug-thres-512000kbytes/resnet20-n4-bs128-lr0.1000-ns1-ds1.0/gpu23-0.log',
    'Overlap': '/home/comp/amelieczhou/DDP-Train/logs/03-21-19:34overlapallreduce-comp-topk-gwarmup-dc1-model-debug-thres-512000kbytes/resnet20-n4-bs128-lr0.1000-ns1-ds0.01/gpu23-0.log',
    'Non-overlap': '/home/comp/amelieczhou/DDP-Train/logs/03-21-19:36no-overlapallreduce-comp-topk-gwarmup-dc1-model-debug-thres-512000kbytes/resnet20-n4-bs128-lr0.1000-ns1-ds0.01/gpu23-0.log',
    #'warmup-overlap': '/home/comp/amelieczhou/DDP-Train/logs/03-21-15:10allreduce-comp-topk-gwarmup-dc1-model-debug-thres-512000kbytes/resnet20-n4-bs128-lr0.1000-ns1-ds0.01/gpu23-0.log',
    #'warmup-non-overlap': '/home/comp/amelieczhou/DDP-Train/logs/03-21-15:23allreduce-comp-topk-gwarmup-dc1-model-debug-thres-512000kbytes/resnet20-n4-bs128-lr0.1000-ns1-ds0.01/gpu23-0.log',
    'Full-Overlap': '/home/comp/amelieczhou/DDP-Train/logs/03-22-13:09overlapallreduce-comp-topk-gwarmup-dc1-model-debug-thres-512000kbytes/resnet20-n4-bs128-lr0.1000-ns1-ds0.01/gpu23-0.log',
    'Half-Overlap': '/home/comp/amelieczhou/DDP-Train/logs/03-22-14:04overlapallreduce-comp-topk-gwarmup-dc1-model-debug-thres-512000kbytes/resnet20-n4-bs128-lr0.1000-ns1-ds0.01/gpu23-0.log'
    # Add more experiments as needed.
}

# Plotting accuracies and losses for each experiment.
plt.subplot(1, 2, 1)
for experiment_name, log_file_path in log_directories.items():
    epochs, accuracies, _ = parse_metrics(log_file_path)
    plt.plot(epochs, accuracies, label=experiment_name)
plt.title('Training Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Subplot for losses.
plt.subplot(1, 2, 2)
for experiment_name, log_file_path in log_directories.items():
    epochs, _, losses = parse_metrics(log_file_path)
    plt.plot(epochs, losses, label=experiment_name)
plt.title('Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('fixed_training_plots.pdf')

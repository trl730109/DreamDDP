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
    'local-sgd': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/none/05-22-13:59gwarmup-dc1-model-debug/gpu15-0.log',
    'gradient_compressed': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-20-17:01-average-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
    'gradient': '/home/yinyiming/DDP-Train-main/logs/resnet20/none/05-20-18:22gwarmup-dc1-model-debug/gpu15-0.log',
    'local-sgd_compressed': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/topk/05-22-14:28-average-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
    
    
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

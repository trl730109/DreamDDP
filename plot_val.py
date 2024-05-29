#!/home/your_username/miniconda3/envs/DDP/bin/python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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

def parse_metrics_val(log_file):
    epochs, val_accuracies, val_losses = [], [], []
    # Adjusted pattern to match the new line format with validation metrics
    epoch_pattern = re.compile(r'Epoch (\d+), lr: ([0-9.]+), val loss: ([0-9.]+), val top-1 acc: ([0-9.]+), top-5 acc: ([0-9.]+)')
    
    with open(log_file, 'r') as file:
        for line in file:
            match = epoch_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                lr = float(match.group(2))
                val_loss = float(match.group(3))
                val_top_1_acc = float(match.group(4))
                # Assuming you only want to collect the val top-1 acc based on your request
                epochs.append(epoch)
                val_accuracies.append(val_top_1_acc)
                val_losses.append(val_loss)
    return epochs, val_accuracies, val_losses

#TOPK s
# log_directories = {
#     'TOPK': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-13-21:07-average-comp-topk-gwarmup-dc1-model-debug/gpu9-0.log',
#     'TIES': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-13-21:53-ties-comp-topk-gwarmup-dc1-model-debug/gpu9-0.log',
#     'TIES_Max': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-13-22:40-ties_max-comp-topk-gwarmup-dc1-model-debug/gpu9-0.log',
#     'Overlap-2': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-14-09:26-overlap-Scalar-2.0-comp-topk-gwarmup-dc1-model-debug/gpu9-0.log',
#     'Overlap-3': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-14-09:49-overlap-Scalar-3.0-comp-topk-gwarmup-dc1-model-debug/gpu9-0.log',
#     'Overlap-4': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-14-10:39-overlap-Scalar-4.0-comp-topk-gwarmup-dc1-model-debug/gpu9-0.log',

# }
#EFTOPK
log_directories = {
    'local-sgd': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/none/05-22-13:59gwarmup-dc1-model-debug/gpu15-0.log',
    #'gradient_compressed': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-20-17:01-average-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
    #'gradient': '/home/yinyiming/DDP-Train-main/logs/resnet20/none/05-20-18:22gwarmup-dc1-model-debug/gpu15-0.log',
    'localsgd_compressed-average': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/topk/05-22-14:28-average-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
    'localsgd-compressed-ties': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/topk/05-27-09:13-SGD-ties-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
    'pseudo-localsgd-compressed': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/topk/05-27-16:24-SGD-ties-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
       
    
    
}


# Plotting accuracies and losses for each experiment.
plt.subplot(1, 2, 1)
for experiment_name, log_file_path in log_directories.items():
    epochs, accuracies, _ = parse_metrics_val(log_file_path)
    smoothed_accuracies = gaussian_filter1d(accuracies, sigma=4)
    plt.plot(epochs, smoothed_accuracies, label=experiment_name)
plt.title('Validation Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Subplot for losses.
plt.subplot(1, 2, 2)
for experiment_name, log_file_path in log_directories.items():
    epochs, _, losses = parse_metrics_val(log_file_path)
    smoothed_losses = gaussian_filter1d(losses, sigma=4)
    plt.plot(epochs, smoothed_losses, label=experiment_name)
plt.title('Validation Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('./plots/test_acc.pdf')
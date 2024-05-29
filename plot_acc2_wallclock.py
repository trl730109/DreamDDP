#!/home/your_username/miniconda3/envs/DDP/bin/python3
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Function to parse time, accuracy, and loss from a log file.
def parse_metrics(log_file):
    times, accuracies, losses = [], [], []
    # Regex pattern to capture the date, time, epoch, accuracy, and loss
    log_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[dl_trainer.py:\d+\] INFO Epoch (\d+), avg train acc: ([0-9.]+), lr: [0-9.]+, avg loss: ([0-9.]+)')
    
    with open(log_file, 'r') as file:
        lines = file.readlines()
        start_time = None
        for line in lines:
            match = log_pattern.search(line)
            if match:
                time_str, epoch, accuracy, loss = match.groups()
                current_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                if not start_time:
                    start_time = current_time
                epoch_time = (current_time - start_time).total_seconds()
                
                times.append(epoch_time)
                accuracies.append(float(accuracy))
                losses.append(float(loss))
                
    return times, accuracies, losses

# Directories containing experiment logs.
log_directories = {
    'local-sgd': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/none/05-22-13:59gwarmup-dc1-model-debug/gpu15-0.log',
    'gradient_compressed': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-20-17:01-average-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
    'gradient': '/home/yinyiming/DDP-Train-main/logs/resnet20/none/05-20-18:22gwarmup-dc1-model-debug/gpu15-0.log',
    'local-sgd_compressed': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/topk/05-22-14:28-average-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
    'pseudo-localsgd-compressed': '/home/yinyiming/DDP-Train-main/localsgd_logs/resnet20/topk/05-27-16:24-SGD-ties-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log'
}

# Plotting accuracies and losses for each experiment.
plt.subplot(1, 2, 1)
for experiment_name, log_file_path in log_directories.items():
    times, accuracies, _ = parse_metrics(log_file_path)
    plt.plot(times, accuracies, label=experiment_name)
plt.title('Training Accuracy vs. Wall Clock Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Accuracy')
plt.legend()

# Subplot for losses.
plt.subplot(1, 2, 2)
for experiment_name, log_file_path in log_directories.items():
    times, _, losses = parse_metrics(log_file_path)
    plt.plot(times, losses, label=experiment_name)
plt.title('Training Loss vs. Wall Clock Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('./plots/training_plots_wallclock.pdf')

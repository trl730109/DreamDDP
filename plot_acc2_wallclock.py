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
    #'localsgd': '/home/comp/amelieczhou/DDP-Train/baselines/localsgd_logs/resnet20/none/06-12-09:33gwarmup-dc1-model-debug-desync-SGD-average-mu_1.0-std_0.5/gpu22-0.log',
    #'Layerwise': '/home/comp/amelieczhou/DDP-Train/test/layerwise/resnet20/none/06-13-16:21-layerwise-SGD-average-mu_0.0-std_0.01/gpu22-0.log',
    #'Sequential':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-16-17:23-seq-SGD-average-mu_0.0-std_0.01/gpu22-0.log',
    'localsgd-20': '/home/comp/amelieczhou/DDP-Train/test/localsgd/resnet20/none/06-18-18:20-localsgd-SGD-average-mu_0.0-std_0.01/gpu23-0.log',
    'Seq-20-fw':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-18-18:49-seq-SGD-average-mu_0.0-std_0.01/gpu23-0.log',
    #'Seq-20-ties_max':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-18-10:15-seq-SGD-ties_max-mu_0.0-std_0.01/gpu22-0.log',
    'Seq-20-ties':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-18-19:18-seq-SGD-ties-mu_0.0-std_0.01/gpu23-0.log',
    #'Seq-2p-ties-fw':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-18-11:01-seq-SGD-ties-mu_0.0-std_0.01/gpu22-0.log',
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
plt.savefig('./plots/training_wallclock.pdf')

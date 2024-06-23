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
    'localsgd': '/home/comp/amelieczhou/DDP-Train/test/localsgd/resnet20/06-21-18:36-localsgd/gpu22-0.log',
    'pipe': '/home/comp/amelieczhou/DDP-Train/test/pipeline/resnet20/06-21-18:46-pipe/gpu22-0.log',
    'sgd':'/home/comp/amelieczhou/DDP-Train/test/sgd/resnet20/06-21-18:26-sgd/gpu22-0.log',

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
plt.savefig('./new_plots/training_wallclock.pdf')

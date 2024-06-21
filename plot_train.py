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
# log_directories = {
#     'SGD': '/home/yinyiming/DDP-Train/logs/resnet20/none/05-30-14:06gwarmup-dc1-model-debug-SGD-average/gpu15-0.log',
#     #'gradient_compressed': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-20-17:01-average-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
#     #'gradient': '/home/yinyiming/DDP-Train-main/logs/resnet20/none/05-20-18:22gwarmup-dc1-model-debug/gpu15-0.log',
#     'Adam': '/home/yinyiming/DDP-Train/logs/resnet20/none/05-30-13:50gwarmup-dc1-model-debug-Adam-average/gpu15-0.log',
#     'AdamW': '/home/yinyiming/DDP-Train/logs/resnet20/none/05-30-14:21gwarmup-dc1-model-debug-AdamW-average/gpu15-0.log',

# }

# log_directories = {
#     #'localsgd': '/home/comp/amelieczhou/DDP-Train/baselines/localsgd_logs/resnet20/none/06-12-09:33gwarmup-dc1-model-debug-desync-SGD-average-mu_1.0-std_0.5/gpu22-0.log',
#     #'Layerwise': '/home/comp/amelieczhou/DDP-Train/test/layerwise/resnet20/none/06-13-16:21-layerwise-SGD-average-mu_0.0-std_0.01/gpu22-0.log',
#     #'Sequential':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-16-17:23-seq-SGD-average-mu_0.0-std_0.01/gpu22-0.log',
#     'localsgd-20': '/home/comp/amelieczhou/DDP-Train/test/localsgd/resnet20/none/06-17-09:54-localsgd-SGD-average-mu_0.0-std_0.01/gpu22-0.log',
#     'Seq-20':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-17-09:30-seq-SGD-average-mu_0.0-std_0.01/gpu22-0.log',
#     #'Seq-20-ties_max':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-18-10:15-seq-SGD-ties_max-mu_0.0-std_0.01/gpu22-0.log',
#     'Seq-20-ties':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-18-10:33-seq-SGD-ties-mu_0.0-std_0.01/gpu22-0.log',
#     'Seq-2p-ties-fw':'/home/comp/amelieczhou/DDP-Train/test/sequential/resnet20/none/06-18-11:01-seq-SGD-ties-mu_0.0-std_0.01/gpu22-0.log',
# }
log_directories = {
    'localsgd': '/home/comp/amelieczhou/DDP-Train/test/localsgd/resnet20/06-21-10:22-localsgd/gpu23-0.log',
    'pipe': '/home/comp/amelieczhou/DDP-Train/test/pipeline/resnet20/06-21-18:06-pipe/gpu22-0.log',
    'sgd':'/home/comp/amelieczhou/DDP-Train/test/sgd/resnet20/06-21-10:33-sgd/gpu23-0.log',

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
plt.savefig('./new_plots/train_acc_comparison.pdf')

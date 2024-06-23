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
    'localsgd': '/home/comp/amelieczhou/DDP-Train/test/localsgd/resnet20/06-23-17:17-localsgd/gpu23-0.log',
    #'pipe': '/home/comp/amelieczhou/DDP-Train/test/pipeline/resnet20/06-21-18:46-pipe/gpu22-0.log',
    'sgd':'/home/comp/amelieczhou/DDP-Train/test/sgd/resnet20/06-21-18:26-sgd/gpu22-0.log',
    'pipe_seq_localsgd_sum':'/home/comp/amelieczhou/DDP-Train/test/pipe_seq_localsgd/resnet20/sum/06-23-16:39-pipe_seq_localsgd/gpu22-0.log',
    'pipe_seq_locasgd_avg': '/home/comp/amelieczhou/DDP-Train/test/pipe_seq_localsgd/resnet20/avg/06-23-16:59-pipe_seq_localsgd/gpu22-0.log'
}


#Add noise
# log_directories = {
#     #'grad_mu_0_std_0.0001': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-07-21:30gwarmup-dc1-model-debug-SGD-average-mu_0.0-std_0.0001/gpu22-0.log',
#     #'grad_mu_0_std_0.001': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-07-21:43gwarmup-dc1-model-debug-SGD-average-mu_0.0-std_0.001/gpu22-0.log',
#     'grad_mu_0_std_0.01': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-07-21:55gwarmup-dc1-model-debug-SGD-average-mu_0.0-std_0.01/gpu22-0.log',
#     'grad_mu_0_std_0.1': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-07-22:07gwarmup-dc1-model-debug-SGD-average-mu_0.0-std_0.1/gpu22-0.log',
#     'grad_mu_0_std_1': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-07-22:19gwarmup-dc1-model-debug-SGD-average-mu_0.0-std_1.0/gpu22-0.log',
#     #'desync_mu_0_std_0.0001': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-08-23:06gwarmup-dc1-model-debug-desync-SGD-average-mu_0.0-std_0.0001/gpu22-0.log',
#     #'desync_mu_0_std_0.001': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-08-23:17gwarmup-dc1-model-debug-desync-SGD-average-mu_0.0-std_0.001/gpu22-0.log',
#     'desync_mu_0_std_0.01': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-08-23:28gwarmup-dc1-model-debug-desync-SGD-average-mu_0.0-std_0.01/gpu22-0.log',
#     'desync_mu_0_std_0.1': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-08-23:39gwarmup-dc1-model-debug-desync-SGD-average-mu_0.0-std_0.1/gpu22-0.log',
#     'desync_mu_0_std_1': '/home/comp/amelieczhou/DDP-Train/logs/resnet20/none/06-08-23:50gwarmup-dc1-model-debug-desync-SGD-average-mu_0.0-std_1.0/gpu22-0.log',
   
    
    #'gradient_compressed': '/home/yinyiming/DDP-Train-main/logs/resnet20/topk/05-20-17:01-average-comp-topk-gwarmup-dc1-model-debug/gpu15-0.log',
    #'gradient': '/home/yinyiming/DDP-Train-main/logs/resnet20/none/05-20-18:22gwarmup-dc1-model-debug/gpu15-0.log',
    # 'Adam': '/home/yinyiming/DDP-Train/logs/resnet20/none/05-30-13:50gwarmup-dc1-model-debug-Adam-average/gpu15-0.log',
    # 'AdamW': '/home/yinyiming/DDP-Train/logs/resnet20/none/05-30-14:21gwarmup-dc1-model-debug-AdamW-average/gpu15-0.log',
    #'localsgd-fixed-lr': '/home/yinyiming/DDP-Train/localsgd_logs/resnet20/none/06-04-09:32gwarmup-dc1-model-debug-SGD-average/gpu15-0.log'

#}


# Plotting accuracies and losses for each experiment.
plt.subplot(1, 2, 1)
for experiment_name, log_file_path in log_directories.items():
    epochs, accuracies, _ = parse_metrics_val(log_file_path)
    smoothed_accuracies = gaussian_filter1d(accuracies, sigma=4)
    plt.plot(epochs, accuracies, label=experiment_name)
plt.title('Validation Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(fontsize='small')

# Subplot for losses.
plt.subplot(1, 2, 2)
for experiment_name, log_file_path in log_directories.items():
    epochs, _, losses = parse_metrics_val(log_file_path)
    smoothed_losses = gaussian_filter1d(losses, sigma=4)
    plt.plot(epochs, losses, label=experiment_name)
plt.title('Validation Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(fontsize='small')

plt.tight_layout()
plt.savefig('./new_plots/test_acc_comparison.pdf')
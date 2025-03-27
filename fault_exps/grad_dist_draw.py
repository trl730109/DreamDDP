import torch

import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Load gradient magnitudes
resnet50_gradients = torch.load('resnet50_gradients.pth')
# gpt2_gradients = torch.load('gpt2_gradients.pth')



def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                            ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

def set_scientific_x_ticks(ax):
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# Function to plot gradient magnitudes
def plot_gradients(gradients, title):
    for iteration, layers in gradients.items():
        for name, grad in layers.items():
            magnitudes = grad.flatten().cpu().numpy()

            fig = plt.figure(figsize=(8, 8))  # Square figure
            # sns.histplot(magnitudes, bins=20, kde=False, color='red', edgecolor='black')
            sns.histplot(magnitudes, bins=50, kde=True, color='red', edgecolor='black')

            ax = fig.gca()
            update_fontsize(ax, 20)
            set_scientific_x_ticks(ax)  # Set x-axis to scientific notation

            plt.xlabel('Bias', fontsize=20)
            plt.ylabel('Frequency', fontsize=20)
            plt.grid(True)
            plt.tight_layout(pad=0)  # Remove margins

            plt.xlabel('Gradient Magnitude')
            plt.ylabel('Frequency')
            plt.show()
            plt.savefig(f'{title}-{iteration}-{name}.pdf')
            plt.savefig(f'{title}-{iteration}-{name}.png')



# Plot ResNet50 gradients
plot_gradients(resnet50_gradients, 'ResNet50')

# # Plot GPT-2 gradients
# plot_gradients(gpt2_gradients, 'GPT-2')





















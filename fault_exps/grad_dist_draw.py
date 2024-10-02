import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load gradient magnitudes
with open('resnet50_gradients.json', 'r') as f:
    resnet50_gradients = json.load(f)

with open('gpt2_gradients.json', 'r') as f:
    gpt2_gradients = json.load(f)

# Function to plot gradient magnitudes
def plot_gradients(gradients, title):
    for iteration, layers in gradients.items():
        magnitudes = list(layers.values())
        sns.histplot(magnitudes, kde=True)
        plt.title(f'{title} - Iteration {iteration}')
        plt.xlabel('Gradient Magnitude')
        plt.ylabel('Frequency')
        plt.show()

# Plot ResNet50 gradients
plot_gradients(resnet50_gradients, 'ResNet50')

# Plot GPT-2 gradients
plot_gradients(gpt2_gradients, 'GPT-2')

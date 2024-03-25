import matplotlib.pyplot as plt
import numpy as np

def plot_tensor_distribution(tensor, epoch):
    # Calculate the distribution
    unique, counts = np.unique(tensor, return_counts=True)
    distribution = counts / len(tensor)

    # Plot
    plt.figure(figsize=[10,6])
    plt.bar(unique, distribution, color='skyblue')
    plt.title('Distribution of indexes in the Tensor')
    plt.xlabel('Number')
    plt.ylabel('Distribution')
    plt.xticks(unique)
    plt.grid(axis='y')

    # Save the plot
    plt.savefig(f'./index_plot/index_distribution_epoch_{epoch}.pdf')
    plt.close() 

def plot_sign_distribution(sign_tensor, epoch):
    # Separate positive, negative, and zero counts
    sign_tensor = sign_tensor.cpu()
    pos_counts = np.maximum(sign_tensor, 0)
    neg_counts = np.abs(np.minimum(sign_tensor, 0))
    zero_counts = (sign_tensor == 0).int() * len(sign_tensor)

    #zero_counts = (sign_tensor == 0).astype(int) * len(sign_tensor)  # Assume max zero count equals the number of workers

    # X locations for the groups
    ind = np.arange(len(sign_tensor))

    # Plot
    fig, ax = plt.subplots()
    p1 = ax.bar(ind, pos_counts, color='green', edgecolor='white')
    p2 = ax.bar(ind, neg_counts, bottom=pos_counts, color='red', edgecolor='white')
    p3 = ax.bar(ind, zero_counts, bottom=pos_counts + neg_counts, color='blue', edgecolor='white')

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('Counts')
    ax.set_title('Distribution of Sign in Tensor')
    ax.set_xticks(ind)
    ax.set_xticklabels([f'Index {i}' for i in range(len(sign_tensor))])
    ax.legend((p1[0], p2[0], p3[0]), ('Positive', 'Negative', 'Zero'))

    plt.show()
    plt.savefig(f'./sign_plot/sign_distribution_epoch_{epoch}.pdf')
    #plt.savefig('sign_training_plots.pdf')
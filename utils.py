import hashlib
import time
import os
import numpy as np
import scipy.stats as stats
import torch
# from horovod.torch.mpi_ops import allreduce_async_
# from horovod.torch.mpi_ops import synchronize
from collections import defaultdict

import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ('true', 'True'):
        return True
    elif isinstance(v, str) and v.lower() in ('false', 'False'):
        return False
    else:
        return v

def group_layers(layer_names):
    layers_dict = defaultdict(list)
    for name in layer_names:
        parts = name.split('.')
        if 'stage' in parts[0]:
            layer_base = '.'.join(parts[:3]) 
        else:
            if 'conv' in parts[0] or 'bn' in parts[0]:
                layer_base = parts[0]
            else:
                layer_base = '.'.join(parts[:2])
        layers_dict[layer_base].append(name)

    # Convert dictionary to list of lists
    return list(layers_dict.values())

# def AllReduce_L2(score_dict, comm_layer_list):
#     handles = []
#     for name in comm_layer_list:
#         handle = allreduce_async_(score_dict[name], average=True, name=name)
#         handles.append(handle)
#     for handle in handles:
#         synchronize(handle)

def adjust_interval(named_params, interval_list, score_list, increasing_factor, interval):
    # Calculate total model size and total model discrepancy
    total_model_size = 0
    total_discrepancy = 0
    for name in named_params:
        total_model_size += named_params[name].numel()
        total_discrepancy += (named_params[name].numel() * score_list[name])

    # Calculate layer-wise cumulative discrepancy and size ratios
    sorted_layers = sorted(score_list, key=score_list.get)  # Sort layers by discrepancy
    #print(f'Sorted layer values are {sorted_layers}')
    cumulative_discrepancy = 0
    cumulative_size = 0

    # Sort the discrepancies and sizes
    sorted_discrepancies = [score_list[name] for name in sorted_layers]
    #print(f'sorted lists are {sorted_discrepancies}')
    sorted_sizes = [named_params[name].numel() for name in sorted_layers]

    # Calculate cumulative values and adjust intervals
    for idx, name in enumerate(sorted_layers):
        cumulative_discrepancy += sorted_discrepancies[idx] * sorted_sizes[idx]
        cumulative_size += sorted_sizes[idx]

        delta_l = cumulative_discrepancy / total_discrepancy
        lambda_l = cumulative_size / total_model_size
        #print(f'Delta and Lambda are {delta_l} and {lambda_l}')

        if delta_l < lambda_l:
            #print(f'Increase the {name} exchange interval by 2.')
            interval_list[name] = interval * increasing_factor
        else:
            #print(f'keep the {name} exchange interval originally.')
            interval_list[name] = interval

    return interval_list


def check_sign_conflicts_simplified(tensors):
    encode_sign = lambda x: 4 * (x > 0).long() + 2 * (x < 0).long() + (x == 0).long()
    encoded_tensors = [encode_sign(tensor) for tensor in tensors]

    aggregated_signs = encoded_tensors[0]
    for encoded_tensor in encoded_tensors[1:]:
        aggregated_signs |= encoded_tensor
    #print(aggregated_signs)
    conflicts = aggregated_signs == 6 | (aggregated_signs == 7)
    all_zeros = aggregated_signs == 1 
    no_conflict = (aggregated_signs == 4) | (aggregated_signs == 2) | (aggregated_signs == 3) | (aggregated_signs == 5)

    conflicts = (conflicts.sum().float() / conflicts.numel())
    all_zeros = (all_zeros.sum().float() / all_zeros.numel())
    no_conflict = (no_conflict.sum().float() / no_conflict.numel())
    return conflicts, all_zeros, no_conflict

def ties_avg(tensors):
    total_sum = sum(tensors)
    gamma_m = torch.sign(total_sum)

    disjoint_mean = torch.zeros_like(tensors[0], dtype=torch.float)
    count_matching_signs = torch.zeros_like(tensors[0], dtype=torch.float)
    
    for tensor in tensors:
        mask = (torch.sign(tensor) == gamma_m) & (tensor != 0)  
        disjoint_mean += tensor * mask.float() 
        count_matching_signs += mask.float()  

    count_matching_signs[count_matching_signs == 0] = 1
    disjoint_mean /= count_matching_signs  
    
    return disjoint_mean

def ties_max(tensors):
    total_sum = sum(tensors)
    gamma_m = torch.sign(total_sum)
    
    disjoint_mean = torch.zeros_like(tensors[0], dtype=torch.float)
    count_matching_signs = torch.zeros_like(tensors[0], dtype=torch.float)
    
    for tensor in tensors:
        mask = (torch.sign(tensor) == gamma_m)# & (tensor != 0)  # Match signs, ignore zeros
        disjoint_mean = torch.maximum((tensor * mask.float()).abs(),disjoint_mean)  # Add matching values
           
    return disjoint_mean * gamma_m

def plot_overlap_trends(data,path_prefix):
    title='Trends of Overlap Cases'
    xlabel='Time or Entry Index'
    ylabel='Percentage'
    figsize=(10, 6)
    data_array = np.array(data)
    data_transposed = data_array.T
    plt.figure(figsize=figsize)  # Set the size of the figure
    num_cases = data_transposed.shape[0]  # Number of overlap cases
    for i in range(num_cases):
        plt.plot(data_transposed[i], label=f'Overlap-{i+1}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()  # Show legend with overlap case labels
    path = "./trend/" + path_prefix+ 'overlap_trend.pdf'
    plt.savefig(path)
    # Show the plot
    plt.show()

def find_single_nonzero_indices(gradient_tensors):
    stacked_gradients = torch.stack(gradient_tensors)
    nonzero_mask = stacked_gradients != 0
    nonzero_counts = nonzero_mask.sum(dim=0)
    single_nonzero_indices = (nonzero_counts == 1).nonzero().squeeze()
    return single_nonzero_indices

def nonzero_distribution(gradient_tensors):
    gradient_tensors = [tensor.to("cpu") for tensor in gradient_tensors]
    stacked_gradients = torch.stack(gradient_tensors)
    signs = stacked_gradients.sign()
    nonzero_mask = stacked_gradients != 0
    max_signs = torch.where(nonzero_mask, signs, torch.tensor(float('-inf'), device=signs.device))
    min_signs = torch.where(nonzero_mask, signs, torch.tensor(float('inf'), device=signs.device))
    max_signs = max_signs.max(dim=0).values
    min_signs = min_signs.min(dim=0).values
    same_sign_nonzero = (max_signs == min_signs) & (max_signs != float('-inf')) & (min_signs != float('inf'))
    nonzero_counts = nonzero_mask.sum(dim=0)
    num_workers = len(gradient_tensors)
    distribution = torch.zeros(num_workers, dtype=torch.float) 

    for i in range(1, num_workers + 1):
        distribution[i - 1] = ((nonzero_counts == i) & same_sign_nonzero).float().sum()
    total_elements = (nonzero_counts > 0).float().sum()
    if total_elements == 0:
        return [0.0] * num_workers 

    distribution_percentages = (distribution / total_elements * 100).tolist()
    return distribution_percentages[:-1]

def plot_similarity_trend(similarity_scores, type, path_prefix):
    iterations = list(range(1, len(similarity_scores) + 1))

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, similarity_scores, marker='o', linestyle='-', color='b')

    # Adding titles and labels
    plt.title('Similarity of Compressed Gradients Across Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Similarity')

    # Optionally add a grid
    plt.grid(True)
    path = "./trend/" + path_prefix+ type + 'similarity.pdf'
    plt.savefig(path)
    # Show the plot
    plt.show()



def multi_tensor_similarity(gradient_tensors):
    stacked_gradients = torch.stack(gradient_tensors)
    signs = stacked_gradients.sign()
    nonzero_mask = stacked_gradients != 0

    # Cases where all or none of the elements are non-zero
    all_nonzero = nonzero_mask.all(dim=0)
    all_zero = (~nonzero_mask).all(dim=0)

    # Determine the max and min signs for each element across the tensors
    max_signs = torch.where(nonzero_mask, signs, torch.tensor(float('-inf'), device=signs.device))
    min_signs = torch.where(nonzero_mask, signs, torch.tensor(float('inf'), device=signs.device))
    max_signs = max_signs.max(dim=0).values
    min_signs = min_signs.min(dim=0).values

    # Define the cases based on the conditions
    case2 = ((max_signs == min_signs) & all_nonzero).sum()
    case3 = ((max_signs != min_signs) & all_nonzero).sum()
    case4 = ((max_signs == min_signs) & ~all_nonzero & ~all_zero).sum()
    case5 = ((max_signs != min_signs) & ~all_nonzero & ~all_zero).sum()

    # Calculate totals for the similarity computation
    positive_cases = case2 + case4
    total_cases = case2 + case3 + case4 + case5

    # Compute the similarity and case4 percentage
    similarity = (positive_cases.float() / total_cases.float()) if total_cases > 0 else 0.0
    case4_percentage = (case4.float() / total_cases.float()) if total_cases > 0 else 0.0

    return similarity.item(), case4_percentage.item()

def cal_similarity(values_store,type):
    similarity_list = []
    for i in range(len(values_store)):
        for j in range(i+1, len(values_store)):
            if type == 'sign':
                similarity_list.append(sign_similarity(values_store[i],values_store[j]))
            elif type == 'cosine':
                similarity_list.append(cosine_similarity(values_store[i],values_store[j]))
    return similarity_list
            
def sign_similarity(tensor1, tensor2):
    sign1 = tensor1.sign()
    sign2 = tensor2.sign()
    both_nonzero_same_sign = (sign1 == sign2) & (sign1 != 0)
    both_nonzero_diff_sign = (sign1 != sign2) & (sign1 != 0) & (sign2 != 0)
    one_zero_one_nonzero = ((sign1 == 0) & (sign2 != 0)) | ((sign2 == 0) & (sign1 != 0))
    positive_count = torch.sum(both_nonzero_same_sign | one_zero_one_nonzero).item()
    negative_count = torch.sum(both_nonzero_diff_sign).item()

    if positive_count + negative_count == 0:
        return 0.0  # Avoid division by zero
    similarity = positive_count / (positive_count + negative_count)

    return similarity

def cosine_similarity(vector1, vector2):
    if vector1.is_cuda:
        vector1 = vector1.cpu()
    if vector2.is_cuda:
        vector2 = vector2.cpu()

    vector1_np = vector1.numpy()
    vector2_np = vector2.numpy()

    dot_product = np.dot(vector1_np, vector2_np)

    norm1 = np.linalg.norm(vector1_np)
    norm2 = np.linalg.norm(vector2_np)

    if norm1 == 0 or norm2 == 0:
        return 0
    else:
        cosine_similarity = dot_product / (norm1 * norm2)
    
    return cosine_similarity

def gen_random_id():
    id_ = hashlib.sha256()
    id_.update(str(time.time()))
    return id_.hexdigest()

def create_path(relative_path):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, relative_path)
    if not os.path.isdir(filename):
        try:
            #os.mkdir(filename)
            os.makedirs(filename)
        except:
            pass

def update_fontsize(ax, fontsize=12.):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

def autolabel(rects, ax, label, rotation=90):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_y() + rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.03*height,
            label,
            ha='center', va='bottom', rotation=rotation)

def topk(tensor, k):
    indexes = np.abs(tensor).argsort()[-k:]
    return indexes, tensor[indexes]

def get_approximate_sigma_scale(density):
    sigma_scale = 1
    if density > 0.7:
        sigma_scale = 0.5
    elif density <= 0.7 and density > 0.05:
        sigma_scale = 1.5
    elif density <= 0.05 and density > 0.01:
        sigma_scale = 2.0
    else:
        sigma_scale = 3.0
    return sigma_scale



def force_insert_item(d, key, val):
    if key not in d:
        d[key] = []
    d[key].append(val)


s=2.18896957e-10 #P102-100
#s=4.99671953e-10 #V100
#a=0.002661810655986525 # small message <1M
#b=1.3644874178760432e-08 # small message <1M
GbE_multi_p_ab_small = {
        2: (1.6e-3, 1.0e-8),
        4: (2.7e-3, 1.3e-8),
        8: (4.0e-3, 1.5e-8),
        #16: (1.1e-2, 1.7e-8)
        16: (1.7e-3, 1.7e-8) #  ImageNet
        #16: (0.05e-2, 0.28e-8) # Inceptionv4 8 layers
        }


GbE_multi_p_ab_large = {
        2: (4.4e-3, 5.8e-9),
        4: (5.6e-3, 7.4e-9),
        8: (7.68e-3, 8.2e-9),
        16: (2.1e-3, 1.7e-8) # good for imagenet
        }

tenGbE_multi_p_ab = {
        2: (1.5e-5, 5.7e-11),
        4: (3.6e-5, 1.1e-10),
        8: (8.5e-5, 1.4e-10),
        16: (1.4e-4, 2.0e-10)
        }



#a=0.015890215705869848 # large message >1M
#b=8.594593687256138e-09 # large message >1M

def topk_perf_model(x, s=s):
    """
    x is the number of parameters
    Return: s * x * log2(x)
    """
    if x == 0.0:
        return 0.0
    return s * x * np.log2(x)

def allgather_perf_model(x, P, density=0.001, eth='GbE'):
    """
    x is the number of parameters
    Return: t = a + b * x
    """
    if x == 0:
        return 0.0
    size = x * P * 4 * density
    if size >= 1024*1024:
        multi_p_ab = GbE_multi_p_ab_large
    else:
        multi_p_ab = GbE_multi_p_ab_small
    a, b = multi_p_ab[P]
    return (a + b * size) * 2

def predict_density_with_size_and_computation(m, comp_time, P):
    alpha = 4*0.436e-3
    beta =  4*9e-6*1e-3
    def _denseallreduce_model(P, m):
        return 2*(P-1)*alpha + 2* (P-1)/P * m * beta

    def _sparseallreduce_model(P, m, rho=0.001):
        return np.log2(P) + 2 * (P - 1) * rho * m * beta

    def _proper_rho_with_sparse_allreduce(P, m, comp_time):
        rho = 0.001
        t = comp_time - np.log2(P) * alpha 
        if t <= 0:
            return rho 
        rho = t/ (2*(P-1)*beta*m)
        if rho > 1.0:
            rho = 0.05
        rho = max(rho, 0.001)
        return rho
    return 0.001
    #if m >= 1024*16:
    #    return 0.001
    #else:
    #    return 1

    #dense_time = _denseallreduce_model(P, m)
    #density = 1
    #if dense_time < comp_time:
    #    return density
    #else:
    #    return _proper_rho_with_sparse_allreduce(P, m, comp_time)

def predict_allreduce_time_with_size(alpha, beta, size, P):
    if size == 0:
        return 0.0
    return alpha + beta * size 

def gen_threshold_from_normal_distribution(p_value, mu, sigma):
    zvalue = stats.norm.ppf((1-p_value)/2)
    return mu+zvalue*sigma, mu-zvalue*sigma

def check_unique(l):
    d = {}
    for k in l:
        if k in d:
            print('element: %s is duplicate in %s' % (k, l))
            return False
        d[k] = 1
    return True


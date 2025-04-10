import numpy as np
from A6000_time import A6000_llama_comm_dict, A6000_llama_bp_dict, A6000_resnet18_comm_dict, A6000_resnet18_bp_dict, A6000_resnet50_comm_dict, A6000_resnet50_bp_dict, A6000_gpt_comm_dict, A6000_gpt_bp_dict
from A800_time import A800_gpt_comm_dict, A800_gpt_bp_dict, A800_llama_comm_dict, A800_llama_bp_dict, A800_resnet18_comm_dict, A800_resnet18_bp_dict, A800_resnet50_comm_dict, A800_resnet50_bp_dict
from H100_time import H100_gpt_comm_dict, H100_gpt_bp_dict, H100_resnet18_comm_dict, H100_resnet18_bp_dict, H100_resnet50_comm_dict, H100_resnet50_bp_dict, H100_llama_comm_dict, H100_llama_bp_dict
#Dreamddp step1: determine the communication schedule
time_dict = {"A6000": {"llama": {"bp": A6000_llama_bp_dict, "comm": A6000_llama_comm_dict}, 
                      "resnet18": {"bp": A6000_resnet18_bp_dict, "comm": A6000_resnet18_comm_dict}, 
                      "resnet50": {"bp": A6000_resnet50_bp_dict, "comm": A6000_resnet50_comm_dict}, 
                      "gpt": {"bp": A6000_gpt_bp_dict, "comm": A6000_gpt_comm_dict}},
             "A800": {"llama": {"bp": A800_llama_bp_dict, "comm": A800_llama_comm_dict}, 
                     "resnet18": {"bp": A800_resnet18_bp_dict, "comm": A800_resnet18_comm_dict}, 
                     "resnet50": {"bp": A800_resnet50_bp_dict, "comm": A800_resnet50_comm_dict}, 
                     "gpt": {"bp": A800_gpt_bp_dict, "comm": A800_gpt_comm_dict}},
             "H100": {"llama": {"bp": H100_llama_bp_dict, "comm": H100_llama_comm_dict}, 
                     "resnet18": {"bp": H100_resnet18_bp_dict, "comm": H100_resnet18_comm_dict}, 
                     "resnet50": {"bp": H100_resnet50_bp_dict, "comm": H100_resnet50_comm_dict}, 
                     "gpt": {"bp": H100_gpt_bp_dict, "comm": H100_gpt_comm_dict}}}

def determine_comm_schedule(comm_dict, bp_dict, H):
    layers = list(comm_dict.keys())
    num_layers = len(layers)
    layers.reverse()
    dp_cache = {}

    def calculate_wait_time(index):
      wait_time = 0
      bp_index = index + 1
      rest = 0
      tmp_idx = 0
      for i in range(index,num_layers):
        left_bp_time = get_remaining_bptime(rest, bp_index)

        if(i == bp_index):
          bp_index += 1
          rest = 0
          left_bp_time = get_remaining_bptime(rest, bp_index)
        if comm_dict[layers[i]] >= left_bp_time:
          tmp_idx = i
          wait_time = comm_dict[layers[i]] - left_bp_time
          break
        else:
          fall_idx, rest = find_fallin(rest, i, bp_index)
          #print(f'fall_idx:{fall_idx}. rest:{rest}')
          bp_index = fall_idx
          continue

      for j in range(tmp_idx + 1, num_layers):
        wait_time += comm_dict[layers[j]]
      return wait_time

    def get_remaining_bptime(rest, index):
      if index == num_layers:
        return 0
      bp_sum = 0
      # print(f'index:{index} num_layers:{num_layers} rest:{rest}')
      for i in range(index, num_layers):
          # print(f'Available layers: {layers[i]} index:{i} bp_time:{bp_dict[layers[i]]}')
          bp_sum += bp_dict[layers[i]]

      if rest == 0:
        return bp_sum
      else:
        return bp_sum - (bp_dict[layers[index]]-rest)

    def find_fallin(rest, index, bp_index):
      if comm_dict[layers[index]] <= rest:
        return bp_index, rest-comm_dict[layers[index]]
      else:
        fall_idx = 0
        rest_time = 0
        if rest == 0:
          total = comm_dict[layers[index]]
          #print(f'Total left: {total}')
          sum = 0
          left_in_bp = 0
          for i in range(bp_index, num_layers):
            sum += bp_dict[layers[i]]
            if (total <= sum):
              fall_idx = i
              rest_time = sum-total
              if(rest_time == 0):
                fall_idx += 1
              break
        else:
          total = comm_dict[layers[index]] - rest
          #print(f'Total left: {total}')
          sum = 0
          left_in_bp = 0
          for i in range(bp_index+1, num_layers):
            sum += bp_dict[layers[i]]
            if (total <= sum):
              fall_idx = i
              rest_time = sum-total
              if(rest_time == 0):
                fall_idx += 1
              break
        return fall_idx, rest_time

# This function will be called in each iteration
    def find_optimal_comm(index, assigned_iterations, left_iterations):
        total_iterations = 1
        current_iteration = assigned_iterations
        total_wait_time = 0
        total_comm_num = 0
        assign_dict = {}
        bp_idx = index + 1
        #comm_idx = index
        rest = 0
        findex = 0
        # base case
        if(index == num_layers):
          return 0,{},0

        if left_iterations == 1:
          for i in range(index, num_layers):
            assign_dict[layers[i]] = current_iteration
          total_wait_time += calculate_wait_time(index)
          return total_wait_time, assign_dict, 1

        for i in range(index, num_layers):
          #print()
          #print(f'current i: {i}. bp index {bp_idx}')
          if i == bp_idx:
            bp_idx += 1

          #print(f'rest:{rest} bp+idx{bp_idx}')
          remaining = get_remaining_bptime(rest,bp_idx)
          #print(f'Remaining time:{remaining}')
          if (comm_dict[layers[i]]) > remaining:
            # case 1: exceed
            if total_comm_num == 0:
              wait_time = comm_dict[layers[i]] - remaining
              # print(f'bp_idx {bp_idx} Comm time: {comm_dict[layers[i]]} remain time {remaining} watitime{wait_time}')
              assign_dict[layers[i]] = current_iteration
              total_wait_time += wait_time
              #print(f'Successfully communicate layer {i} Total wait time: {total_wait_time}')
              # process the next layer in next iteration
              wait_time1, dict1, iter1= find_optimal_comm(i+1, current_iteration+1,left_iterations-1 )
              total_wait_time += wait_time1
              total_iterations += iter1
              for name in dict1.keys():
                assign_dict[name] = dict1[name]
              break
            else:
              wait_time = comm_dict[layers[i]] - remaining
              #print(f'Enterin the first subloop')
              wait_time1, dict1, iter1= find_optimal_comm(i+1, current_iteration+1, left_iterations -1 )
              #print(f'first subloop total wait time: {wait_time + wait_time1}')
              #print(f'Enterin the second subloop')
              wait_time2, dict2, iter2 = find_optimal_comm(i,current_iteration+1, left_iterations -1)
              #print(f'first subloop total wait time: {wait_time2}')
              if (wait_time + wait_time1 <= wait_time2):
                assign_dict[layers[i]] = current_iteration
                total_wait_time += (wait_time + wait_time1)
                total_iterations += iter1
                for name in dict1.keys():
                  assign_dict[name] = dict1[name]
              else:
                total_wait_time += wait_time2
                total_iterations += iter2
                for name in dict2.keys():
                  assign_dict[name] = dict2[name]
              break
          else:
            bp_idx, rest = find_fallin(rest, i, bp_idx)
            #print(f'current layer {i} comm falls in {bp_idx}. rest time: {rest}')
            assign_dict[layers[i]] = current_iteration
            total_comm_num += 1
            #print(f'Successfully communicate layer {i} within the duration')

        return total_wait_time, assign_dict, total_iterations



    total_wait_time, optimal_schedule, total_iterations = find_optimal_comm(0, 0,H)
    return total_wait_time, optimal_schedule, total_iterations


# dreamddp step2: fillin more layers for the bubbles
def fillin_more_layers(total_iterations, comm_dict, bp_dict, schedule):
  new_schedule = {}
  layers = list(comm_dict.keys())
  num_layers = len(layers)
  layers.reverse()
  for name in layers:
    new_schedule[name] = []
    new_schedule[name].append(schedule[name])

  def get_remaining_bptime(rest, index, des):
      if index == des + 1:
        return 0
      bp_sum = 0
      # print(f'index:{index} num_layers:{num_layers} rest:{rest}')
      for i in range(index, des+1):
          # print(f'Available layers: {layers[i]} index:{i} bp_time:{bp_dict[layers[i]]}')
          bp_sum += bp_dict[layers[i]]

      if rest == 0:
        return bp_sum
      else:
        return bp_sum - (bp_dict[layers[index]]-rest)

  def find_first_comm_layer(iteration):
    for index, name in enumerate(schedule.keys()):
      if schedule[name] == iteration:
        return index,name
  def find_fallin(rest, index, bp_index):
      #print(f'rest:{rest} index:{index} bp_index:{bp_index}')
      #print(f'Comm time:{comm_dict[layers[index]]} rest {rest}')
      if comm_dict[layers[index]] <= rest:
        return bp_index, rest-comm_dict[layers[index]]
      else:
        fall_idx = 0
        rest_time = 0
        if rest == 0:
          total = comm_dict[layers[index]]
          #print(f'Total left: {total}')
          sum = 0
          left_in_bp = 0
          for i in range(bp_index, num_layers):
            sum += bp_dict[layers[i]]
            if (total <= sum):
              fall_idx = i
              rest_time = sum-total
              if(rest_time == 0):
                fall_idx += 1
              break
        else:
          total = comm_dict[layers[index]] - rest
          #print(f'Total left: {total}')
          sum = 0
          left_in_bp = 0
          for i in range(bp_index+1, num_layers):
            sum += bp_dict[layers[i]]
            if (total <= sum):
              fall_idx = i
              rest_time = sum-total
              if(rest_time == 0):
                fall_idx += 1
              break
        return fall_idx, rest_time

  for iteration in range(total_iterations):
    # print()
    # print(f'Iteration: {iteration}')
    des,name = find_first_comm_layer(iteration)
    # print(f'First comm layer:{des} iteration:{iteration}')
    rest = 0
    bp_index = 1
    if(des == 0):
      print(f'No available position in the first iteration')
      continue
    for i in range(des):
      if (bp_index == i):
        # print('Add bp index')
        bp_index += 1
        rest = 0
      # print(f'comm index:{i} bp_index:{bp_index}')
      remaining = get_remaining_bptime(rest,bp_index,des)
      # print(f'Remaining time: {remaining}')
      if comm_dict[layers[i]] <= remaining:
        fall_idx, rest_time = find_fallin(rest,i,bp_index)
        bp_index = fall_idx
        rest = rest_time
        new_schedule[layers[i]].append(iteration)
        # print(f'Successfully fillin {layers[i]} index: {i}')
      else:
        break
  return new_schedule


''' Split LocalSGD's communication evenly into each iteration. At the same time, overlap layer comm with bp.
for example: totally 4 layers, communicating in 2 iterations. L4, L3 are communicated in iteration 1. L2 and L1 are communicated in iteration 2.
'''
def pipe_seq_localsgd_waittime(bp_dict, comm_dict,H):
  wait_list = []
  layers = list(comm_dict.keys())
  num_layers = len(layers)
  layers.reverse()
  if (len(layers) % H) == 0:
    layers_per_iter = int(len(layers) / H)
  else:
    layers_per_iter = int(len(layers) / H) + 1
  def find_fallin(rest, index, bp_index):
      #print(f'rest:{rest} index:{index} bp_index:{bp_index}')
      #print(f'Comm time:{comm_dict[layers[index]]} rest {rest}')
      if comm_dict[layers[index]] <= rest:
        return bp_index, rest-comm_dict[layers[index]]
      else:
        fall_idx = 0
        rest_time = 0
        if rest == 0:
          total = comm_dict[layers[index]]
          #print(f'Total left: {total}')
          sum = 0
          left_in_bp = 0
          for i in range(bp_index, num_layers):
            sum += bp_dict[layers[i]]
            if (total <= sum):
              fall_idx = i
              rest_time = sum-total
              if(rest_time == 0):
                fall_idx += 1
              break
        else:
          total = comm_dict[layers[index]] - rest
          #print(f'Total left: {total}')
          sum = 0
          left_in_bp = 0
          for i in range(bp_index+1, num_layers):
            sum += bp_dict[layers[i]]
            if (total <= sum):
              fall_idx = i
              rest_time = sum-total
              if(rest_time == 0):
                fall_idx += 1
              break
        return fall_idx, rest_time
  def get_remaining_bptime(rest, index):
      if index == num_layers:
        return 0
      bp_sum = 0
      # print(f'index:{index} num_layers:{num_layers} rest:{rest}')
      for i in range(index, num_layers):
          # print(f'Available layers: {layers[i]} index:{i} bp_time:{bp_dict[layers[i]]}')
          bp_sum += bp_dict[layers[i]]

      if rest == 0:
        return bp_sum
      else:
        return bp_sum - (bp_dict[layers[index]]-rest)
  def get_waittime(start_index, end_index):
    bp_idx = start_index + 1
    total_wait_time = 0
    rest = 0
    for i in range(start_index, end_index):
      if (i == bp_idx):
        bp_idx += 1
        rest = 0
      remaining = get_remaining_bptime(rest,bp_idx)
      if(comm_dict[layers[i]] >= remaining):
        wait_time = comm_dict[layers[i]] - remaining
        total_wait_time += wait_time
        for j in range(i+1, end_index):
          total_wait_time += comm_dict[layers[j]]
        return total_wait_time
      else:
        fall_idx, rest_time = find_fallin(rest, i, bp_idx)
        bp_idx = fall_idx
        rest = rest_time
    return total_wait_time

  for i in range(H):
    start_index = layers_per_iter * i
    end_index = min(start_index + layers_per_iter, len(layers))
    # print(f'start index:{start_index} end_index:{end_index}')
    wt = get_waittime(start_index, end_index)
    wait_list.append(wt)
  return wait_list
  #waitlist contains the time not covered by bp process in each iteration, sum it to get the whole extra communication time
  

def get_time(comm_dict, bp_dict, H):
  def round_dict_values(input_dict, decimal_places):
    return {key: round(value, decimal_places) for key, value in input_dict.items()}

  rounded_comm_dict = round_dict_values(comm_dict, 8)
  rounded_bp_dict = round_dict_values(bp_dict, 8)
  #calculate the extra communication time for partial localsgd
  plsgd_wait_list = pipe_seq_localsgd_waittime(rounded_bp_dict, rounded_comm_dict,H)
  #determine the communication schedule for dreamddp
  total_wait_time, schedule, total_iterations = determine_comm_schedule(rounded_comm_dict, rounded_bp_dict, H)
  return plsgd_wait_list, [total_wait_time, schedule, total_iterations], H


def cal_sgd(total_iterations, train_time, comm_time):
  time_sum = 0
  time_per_iter = train_time + comm_time
  for i in range(total_iterations):
    time_sum += time_per_iter
  return time_sum / total_iterations

def cal_pipe_sgd(total_iterations, train_time, comm_time):
  time_sum = 0
  time_per_iter = train_time + comm_time * 0.6
  for i in range(total_iterations):
    time_sum += time_per_iter
  return time_sum / total_iterations

def cal_localsgd(total_iterations,nsteps_localsgd, train_time, comm_time):
  time_sum = 0

#   train_time = time_dict[dnn][nworkers]['train']
#   comm_time = time_dict[dnn][nworkers]['comm']
  for i in range(total_iterations):
    if(i != 0 and i % nsteps_localsgd == 0):
      time_sum += (train_time + comm_time)
    else:
      time_sum += train_time

  return time_sum / total_iterations


def cal_dreamddp(total_iterations, H, waittime_per_H, train_time):
  time_sum = 0

  for i in range(total_iterations):
    if(i != 0 and i % H == 0):
      time_sum += (train_time + waittime_per_H)
    else:
      time_sum += train_time
  return time_sum / total_iterations

def cal_pipe_seq_localsgd(total_iterations, H, wait_list, train_time):
  time_dict = {}
  time_sum = 0
#   train_time = time_dict[dnn][nworkers]['train']
#   comm_time = time_dict[dnn][nworkers]['comm']
  for i in range(total_iterations):
    time_sum += (train_time + wait_list[i%H])

  return time_sum / total_iterations

# def main():
#     pass
def scale_comm_time(comm_dict, current_bandwidth, expected_bandwidth):
    return  {key: value * current_bandwidth / expected_bandwidth for key, value in comm_dict.items()}

def round_dict_values(input_dict, decimal_places):
    return {key: round(value, decimal_places) for key, value in input_dict.items()}

if __name__ == "__main__":
    # servers = ['A6000', 'A800', 'H100']
    # models = ['llama', 'gpt', 'resnet18', 'resnet50']
    servers = ['A6000', 'A800', 'H100']
    models = ['llama', 'gpt', 'resnet18', 'resnet50']
    iterations = 1000
    nsteps_localsgd = 10
    expected_bandwidth = 10
    
    # 创建输出文件
    with open('output.txt', 'w') as f:
        for server in servers:
            for model in models:
                current_bandwidth = 10
                output_str = f'Server: {server}, Model: {model}, Current Bandwidth: {current_bandwidth}\n'
                
                # comm_dict = round_dict_values(scale_comm_time(time_dict[server][model]['comm'], current_bandwidth, expected_bandwidth), 8)
                # bp_dict = round_dict_values(time_dict[server][model]['bp'], 8)
                comm_dict = scale_comm_time(time_dict[server][model]['comm'], current_bandwidth, expected_bandwidth)
                bp_dict = time_dict[server][model]['bp']
                
                total_comm_time = sum(comm_dict.values())
                total_bp_time = sum(bp_dict.values())
                
                mean_comm_time = total_comm_time / len(comm_dict)
                mean_bp_time = total_bp_time / len(bp_dict)
                output_str += f'total communication time: {total_comm_time}\n'
                output_str += f'total bp time: {total_bp_time}\n'
                
                output_str += f'total layers: {len(comm_dict)}\n'
                output_str += f'mean layer communication time: {mean_comm_time}\n'
                output_str += f'mean layer bp time: {mean_bp_time}\n'
                
                plsgd_wait_list, [total_wait_time, schedule, total_iterations], nsteps_localsgd = get_time(comm_dict, bp_dict, nsteps_localsgd)
                #calculate the extra communication time for partial localsgd and dreamddp
                partial_localsgd_extra_comm_time = sum(plsgd_wait_list)
                dreamddp_extra_comm_time = total_wait_time
                
                sgd_time_per_iter = cal_sgd(iterations, total_bp_time, total_comm_time)
                pipe_sgd_time_per_iter = cal_pipe_sgd(iterations, total_bp_time, total_comm_time)
                localsgd_time_per_iter = cal_localsgd(iterations, nsteps_localsgd, total_bp_time, total_comm_time)
                dreamddp_time_per_iter = cal_dreamddp(iterations, nsteps_localsgd, dreamddp_extra_comm_time, total_bp_time)
                pipe_seq_localsgd_time_per_iter = cal_pipe_seq_localsgd(total_iterations, nsteps_localsgd, plsgd_wait_list, total_bp_time)
                
                output_str += f'----------------------- Final results -----------------------\n'
                output_str += f'SGD time per iter: {sgd_time_per_iter}\n'
                output_str += f'Pipe SGD time per iter: {pipe_sgd_time_per_iter}\n'
                output_str += f'Localsgd time per iter: {localsgd_time_per_iter}\n'
                output_str += f'Dreamddp time per iter: {dreamddp_time_per_iter}\n'
                output_str += f'Pipe seq Localsgd time per iter: {pipe_seq_localsgd_time_per_iter}\n\n'
                
                # 写入文件
                f.write(output_str)
                
                # 同时打印到控制台
                print(output_str)

import json
import os
import argparse
import numpy as np
from pathlib import Path
from cal_alg_time import *

def load_json_from_dir(base_dir):
    base_dir = Path(base_dir)
    bp_dir = base_dir / 'bp'
    comm_dir = base_dir / 'comm'
    
    # 查找 bp JSON 文件
    bp_files = list(bp_dir.glob('bp_*.json'))
    if not bp_files:
        raise FileNotFoundError(f"未找到 BP JSON 文件在目录: {bp_dir}")
    bp_file = bp_files[0]  # 使用第一个找到的文件
    
    # 查找 comm JSON 文件
    comm_files = list(comm_dir.glob('comm_*.json'))
    if not comm_files:
        raise FileNotFoundError(f"未找到 Comm JSON 文件在目录: {comm_dir}")
    comm_file = comm_files[0]  # 使用第一个找到的文件
    
    print(f"读取 BP 文件: {bp_file}")
    print(f"读取 Comm 文件: {comm_file}")
    
    # 读取 BP JSON
    with open(bp_file, 'r') as f:
        bp_dict = json.load(f)
    
    # 读取 Comm JSON
    with open(comm_file, 'r') as f:
        comm_dict = json.load(f)
    
    # 处理 NaN 值（如果有的话）
    bp_dict = {k: (0.0 if (isinstance(v, float) and (v != v or v == float('inf'))) else v) 
              for k, v in bp_dict.items()}
    comm_dict = {k: (0.0 if (isinstance(v, float) and (v != v or v == float('inf'))) else v) 
                for k, v in comm_dict.items()}
    
    return bp_dict, comm_dict


def cal_sgd(total_iterations, train_time, comm_time):
    """计算标准 SGD 的平均时间"""
    time_sum = 0
    time_per_iter = train_time + comm_time
    for i in range(total_iterations):
        time_sum += time_per_iter
    return time_sum / total_iterations


def cal_pipe_sgd(total_iterations, train_time, comm_time):
    """计算 Pipeline SGD 的平均时间"""
    time_sum = 0
    time_per_iter = train_time + comm_time * 0.6
    for i in range(total_iterations):
        time_sum += time_per_iter
    return time_sum / total_iterations


def cal_localsgd(total_iterations, nsteps_localsgd, train_time, comm_time):
    """计算 LocalSGD 的平均时间"""
    time_sum = 0
    for i in range(total_iterations):
        if(i != 0 and i % nsteps_localsgd == 0):
            time_sum += (train_time + comm_time)
        else:
            time_sum += train_time
    return time_sum / total_iterations


def cal_dreamddp(total_iterations, H, waittime_per_H, train_time):
    """计算 DreamDDP 的平均时间"""
    time_sum = 0
    for i in range(total_iterations):
        if(i != 0 and i % H == 0):
            time_sum += (train_time + waittime_per_H)
        else:
            time_sum += train_time
    return time_sum / total_iterations


def cal_pipe_seq_localsgd(total_iterations, H, wait_list, train_time):
    """计算 Pipeline Sequential LocalSGD 的平均时间"""
    time_sum = 0
    for i in range(total_iterations):
        time_sum += (train_time + wait_list[i % H])
    return time_sum / total_iterations


def calculate_algorithm_times(time_list, total_iterations=1000):
    """
    计算 5 个算法的平均时间和 DreamDDP 的速度提升
    
    Args:
        time_list: [bp_sum, comm_sum, plsgd_wait_list, [wait_time, total_iterations], nsteps_localsgd]
        total_iterations: 用于计算平均时间的迭代次数，默认 1000
    
    Returns:
        tuple: (alg_times_dict, speedups_dict)
            - alg_times_dict: 包含 5 个算法时间的字典
            - speedups_dict: 包含 DreamDDP 相对于 pipe_sgd 和 localsgd 的加速比
    """
    bp_sum = time_list[0]  # train_time
    comm_sum = time_list[1]  # comm_time
    plsgd_wait_list = time_list[2]  # wait_list
    wait_time = time_list[3][0]  # waittime_per_H
    nsteps_localsgd = time_list[4]  # H (nsteps_localsgd)
    
    alg_times = {}
    alg_times["sgd"] = cal_sgd(total_iterations, bp_sum, comm_sum)
    alg_times["pipe_sgd"] = cal_pipe_sgd(total_iterations, bp_sum, comm_sum)
    alg_times["localsgd"] = cal_localsgd(total_iterations, nsteps_localsgd, bp_sum, comm_sum)
    alg_times["pipe_seq_localsgd"] = cal_pipe_seq_localsgd(total_iterations, nsteps_localsgd, plsgd_wait_list, bp_sum)
    alg_times["dream_ddp"] = cal_dreamddp(total_iterations, nsteps_localsgd, wait_time, bp_sum)
    
    # 计算 DreamDDP 相对于 pipe_sgd 和 localsgd 的加速比
    # 公式: (dream_ddp - alg) / dream_ddp
    speedups = {}
    if alg_times["dream_ddp"] > 0:
        if alg_times["pipe_sgd"] > 0:
            speedups["over_pipe_sgd"] = (alg_times["pipe_sgd"]) / alg_times["dream_ddp"]
        else:
            speedups["over_pipe_sgd"] = 0.0
        
        if alg_times["localsgd"] > 0:
            speedups["over_localsgd"] = (alg_times["localsgd"]) / alg_times["dream_ddp"]
        else:
            speedups["over_localsgd"] = 0.0
    else:
        speedups["over_pipe_sgd"] = 0.0
        speedups["over_localsgd"] = 0.0
    
    return alg_times, speedups


def process_model_from_dir(base_dir, H_values=[5, 10], bp_multiplier=1.0, comm_multiplier=1.0):
    """
    从目录读取 JSON 文件并进行调度计算
    
    Args:
        base_dir: 基础目录路径
        H_values: 要测试的 H 值列表，默认 [5, 10]
        bp_multiplier: BP 时间的倍数，默认 1.0
        comm_multiplier: 通信时间的倍数，默认 1.0
    
    Returns:
        dict: 包含所有计算结果的字典
    """
    print("\n" + "="*80)
    print(f"处理数据目录: {base_dir}")
    print("="*80)
    
    try:
        bp_dict, comm_dict = load_json_from_dir(base_dir)
        
        # Apply multipliers if needed
        if bp_multiplier != 1.0:
            bp_dict = {key: value * bp_multiplier for key, value in bp_dict.items()}
            print(f'应用 BP 倍数: {bp_multiplier}')
        if comm_multiplier != 1.0:
            comm_dict = {key: value * comm_multiplier for key, value in comm_dict.items()}
            print(f'应用 Comm 倍数: {comm_multiplier}')
        
        # Calculate sums
        bp_sum = np.sum(list(bp_dict.values()))
        comm_sum = np.sum(list(comm_dict.values()))
        
        print(f'\nBP 总和: {bp_sum}')
        print(f'Comm 总和: {comm_sum}')
        print(f'层数: {len(bp_dict)}')
        
        results = {
            'bp_sum': bp_sum,
            'comm_sum': comm_sum,
            'num_layers': len(bp_dict),
            'H_results': {}
        }
        
        # Run scheduling calculations with different H values
        for H in H_values:
            print(f'\n--- 使用 H={H} 进行计算 ---')
            plsgd_wait_list, [wait_time, schedule, total_iterations], nsteps_localsgd = get_time(
                comm_dict, bp_dict, H
            )
            
            # print(f'scheduling result:H={H}')
            # print(f'schedule: {schedule}')
            scheduling_path = os.path.join(base_dir, 'dreamddp_scheduling.json')
            output = {"H": H, "schedule": schedule}
            with open(scheduling_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, 'item') else x)
            print(f'scheduling result saved to {scheduling_path} (H={H})')
            # print(f'plsgd_wait_list: {plsgd_wait_list}')
            # print(f'wait_time: {wait_time}')
            # print(f'total_iterations: {total_iterations}')
            # print(f'nsteps_localsgd: {nsteps_localsgd}')
            
            time_list = [
                bp_sum, 
                comm_sum, 
                plsgd_wait_list, 
                [wait_time, total_iterations], 
                nsteps_localsgd
            ]
            
            # 计算 5 个算法的时间
            alg_times, speedups = calculate_algorithm_times(time_list, total_iterations=1000)
            
            # print(f'\n时间列表 (H={H}): {time_list}')
            # print(f'总等待时间: {wait_time}')
            # print(f'总迭代次数: {total_iterations}')
            # print(f'\n--- 5 个算法的平均时间 (1000 次迭代) ---')
            # print(f'SGD:                    {alg_times["sgd"]:.6f}')
            # print(f'Pipeline SGD:            {alg_times["pipe_sgd"]:.6f}')
            # print(f'LocalSGD:                {alg_times["localsgd"]:.6f}')
            # print(f'Pipeline Seq LocalSGD:   {alg_times["pipe_seq_localsgd"]:.6f}')
            # print(f'DreamDDP:                {alg_times["dream_ddp"]:.6f}')
            # print(f'\n--- DreamDDP 速度提升 ---')
            # print(f'DreamDDP over Pipeline SGD:  {speedups["over_pipe_sgd"]:.4f}x')
            # print(f'DreamDDP over LocalSGD:      {speedups["over_localsgd"]:.4f}x')
            # print(f'\n调度方案: {schedule}')
            
            # Fill in more layers
            filled_schedule = fillin_more_layers(
                total_iterations, 
                comm_dict, 
                bp_dict, 
                schedule
            )
            # print(f'填充后的调度方案: {filled_schedule}')
            
            results['H_results'][H] = {
                'plsgd_wait_list': plsgd_wait_list,
                'wait_time': wait_time,
                'schedule': schedule,
                'total_iterations': total_iterations,
                'filled_schedule': filled_schedule,
                'time_list': time_list,
                'algorithm_times': alg_times,
                'speedups': speedups
            }
        
        return results
        
    except FileNotFoundError as e:
        print(f'错误: 文件未找到 - {e}')
        return None
    except Exception as e:
        print(f'处理数据时出错: {e}')
        import traceback
        traceback.print_exc()
        return None


def determine_comm_schedule(comm_dict, bp_dict, H):
    layers = list(comm_dict.keys())
    num_layers = len(layers)
    # Sort layers by their backward processing sequence (assuming L1, L2, L3, L4...)
    # layers.sort(key=lambda x: int(x[1:]), reverse=True)
    layers.reverse()
    # print(layers)
    dp_cache = {}

    def calculate_wait_time(index):
      wait_time = 0
      bp_index = index + 1
      rest = 0
      tmp_idx = 0
      for i in range(index,num_layers):
        left_bp_time = get_remaining_bptime(rest, bp_index)
        #print(f'left_bp_time:{left_bp_time}')
        if(i == bp_index):
          #print('Condition satisfied')
          bp_index += 1
          rest = 0
          left_bp_time = get_remaining_bptime(rest, bp_index)
          #print(f'left_bp_time:{left_bp_time}')
        #print(f'i:{i}. bp_index:{bp_index}')
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
  
def fillin_more_layers(total_iterations, comm_dict, bp_dict, schedule):
  # 先对齐 bp / comm 的 key，保证和 schedule 使用同一套 layer 名称
  aligned_bp_dict, aligned_comm_dict = align_bp_comm(bp_dict, comm_dict)

  new_schedule = {}
  layers = list(aligned_comm_dict.keys())
  num_layers = len(layers)
  layers.reverse()
  for name in layers:
    new_schedule[name] = []
    # 有些 layer 可能没有被调度到（不在 schedule），这种保持为空列表即可
    if name in schedule:
      new_schedule[name].append(schedule[name])

  def get_remaining_bptime(rest, index, des):
      if index == des + 1:
        return 0
      bp_sum = 0
      # print(f'index:{index} num_layers:{num_layers} rest:{rest}')
      for i in range(index, des+1):
          # print(f'Available layers: {layers[i]} index:{i} bp_time:{bp_dict[layers[i]]}')
          bp_sum += aligned_bp_dict[layers[i]]

      if rest == 0:
        return bp_sum
      else:
        return bp_sum - (aligned_bp_dict[layers[index]]-rest)

  def find_first_comm_layer(iteration):
    for index, name in enumerate(schedule.keys()):
      if schedule[name] == iteration:
        return index,name
  def find_fallin(rest, index, bp_index):
      #print(f'rest:{rest} index:{index} bp_index:{bp_index}')
      #print(f'Comm time:{aligned_comm_dict[layers[index]]} rest {rest}')
      if aligned_comm_dict[layers[index]] <= rest:
        return bp_index, rest-aligned_comm_dict[layers[index]]
      else:
        fall_idx = 0
        rest_time = 0
        if rest == 0:
          total = aligned_comm_dict[layers[index]]
          #print(f'Total left: {total}')
          sum = 0
          left_in_bp = 0
          for i in range(bp_index, num_layers):
            sum += aligned_bp_dict[layers[i]]
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
            sum += aligned_bp_dict[layers[i]]
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
      if aligned_comm_dict[layers[i]] <= remaining:
        fall_idx, rest_time = find_fallin(rest,i,bp_index)
        bp_index = fall_idx
        rest = rest_time
        new_schedule[layers[i]].append(iteration)
        # print(f'Successfully fillin {layers[i]} index: {i}')
      else:
        break
  return new_schedule

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


def align_bp_comm(bp_dict, comm_dict):
  """
  对齐 bp_dict 和 comm_dict 的 layer 名称。
  主要解决 LoRA 情况下：
    - BP 使用模块名:  ...lora_A.default
    - Comm 使用参数名衍生的层名: ...lora_A
  导致 key 对不上的问题。
  返回: (aligned_bp_dict, aligned_comm_dict)
  """
  aligned_comm_dict = {}
  for k, v in comm_dict.items():
    if k in bp_dict:
      # 完全匹配，直接使用
      aligned_comm_dict[k] = v
    else:
      # 尝试匹配 LoRA 场景:
      # 1) comm: ...lora_A  , bp: ...lora_A.default
      candidate_with_default = k + ".default"
      # 2) comm: ...lora_A.default , bp: ...lora_A （目前暂未出现，但也兼容）
      candidate_without_default = k[:-8] if k.endswith(".default") else None

      if candidate_with_default in bp_dict:
        aligned_comm_dict[candidate_with_default] = v
      elif candidate_without_default and candidate_without_default in bp_dict:
        # 这里要使用 bp_dict 中的 canonical key（去掉 .default），
        # 否则后面用这个 key 去 bp_dict 取值会 KeyError
        aligned_comm_dict[candidate_without_default] = v
      else:
        # 这个 layer 在 BP 中不存在，跳过
        continue

  aligned_bp_dict = {k: bp_dict[k] for k in aligned_comm_dict.keys()}
  return aligned_bp_dict, aligned_comm_dict


def get_time(comm_dict, bp_dict, H):
  """对齐后计算等待时间与调度。"""

  aligned_bp_dict, aligned_comm_dict = align_bp_comm(bp_dict, comm_dict)

  def round_dict_values(input_dict, decimal_places):
    return {key: round(value, decimal_places) for key, value in input_dict.items()}

  rounded_comm_dict = round_dict_values(aligned_comm_dict, 8)
  rounded_bp_dict = round_dict_values(aligned_bp_dict, 8)

  plsgd_wait_list = pipe_seq_localsgd_waittime(rounded_bp_dict, rounded_comm_dict, H)
  total_wait_time, schedule, total_iterations = determine_comm_schedule(rounded_comm_dict, rounded_bp_dict, H)
  return plsgd_wait_list, [total_wait_time, schedule, total_iterations], H


def main():
    parser = argparse.ArgumentParser(description='DreamDDP 调度计算工具')
    parser.add_argument('input_dir', type=str,
                       help='输入目录路径，应包含 bp/ 和 comm/ 子目录，例如: /workspace/tzc/DDP-Train/time/Qwen2.5-1.5B/8')
    parser.add_argument('--H', type=int, default=10,
                       help='要测试的 H 值，默认: 8')
    parser.add_argument('--bp_multiplier', type=float, default=1.0,
                       help='BP 时间的倍数，默认: 1.0')
    parser.add_argument('--comm_multiplier', type=float, default=1.0,
                       help='通信时间的倍数，默认: 1.0')
    
    args = parser.parse_args()
    
    # 确保 H_values 是一个列表
    if isinstance(args.H, int):
        H_values = [args.H]
    else:
        H_values = args.H if isinstance(args.H, list) else [args.H]
    
    # 处理模型数据
    results = process_model_from_dir(
        args.input_dir,
        H_values=H_values,
        bp_multiplier=args.bp_multiplier,
        comm_multiplier=args.comm_multiplier
    )
    
    if results:       # 输出算法时间总结
        # print("\n" + "="*80)
        # print("算法时间总结 (平均时间，1000 次迭代)")
        # print("="*80)
        print(f"{'算法':<25} ", end="")
        for H in H_values:
            print(f"H={H:<15} ", end="")
        print()
        print("-" * 80)
        
        alg_names = ["sgd", "pipe_sgd", "localsgd", "pipe_seq_localsgd", "dream_ddp"]
        alg_display_names = ["SGD", "ASC-WFBP", "FLSGD", "PLSGD", "DreamDDP"]
        
        for alg_name, display_name in zip(alg_names, alg_display_names):
            print(f"{display_name:<25} ", end="")
            for H in H_values:
                if H in results['H_results']:
                    alg_time = results['H_results'][H]['algorithm_times'][alg_name]
                    print(f"{alg_time:<15.6f} ", end="")
                else:
                    print(f"{'N/A':<15} ", end="")
            print()
        
        print("="*80)
        print()
        print("DreamDDP 速度提升总结")
        print(f"{'速度提升':<25} ", end="")
        for H in H_values:
            print(f"H={H:<15} ", end="")
        print()
        
        speedup_names = ["over_pipe_sgd", "over_localsgd"]
        speedup_display_names = ["DreamDDP / Pipeline SGD", "DreamDDP / LocalSGD"]
        
        for speedup_name, display_name in zip(speedup_names, speedup_display_names):
            print(f"{display_name:<25} ", end="")
            for H in H_values:
                if H in results['H_results']:
                    speedup = results['H_results'][H]['speedups'][speedup_name]
                    print(f"{speedup:<15.4f}x ", end="")
                else:
                    print(f"{'N/A':<15} ", end="")
            print()
        
        print("="*80)
        
        # 保存结果到文件
        output_data = {
            'algorithm_times': {},
            'speedups': {}
        }
        
        # 组织数据：按H值组织
        for H in H_values:
            if H in results['H_results']:
                output_data['algorithm_times'][f'H_{H}'] = results['H_results'][H]['algorithm_times']
                output_data['speedups'][f'H_{H}'] = results['H_results'][H]['speedups']
        
        # 保存为JSON格式
        json_output_path = os.path.join(args.input_dir, 'scheduling_results.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到 JSON 文件: {json_output_path}")
        
        # 保存为TXT格式（更易读）
        txt_output_path = os.path.join(args.input_dir, 'scheduling_results.txt')
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("算法时间总结 (平均时间，1000 次迭代)\n")
            f.write("="*80 + "\n")
            f.write(f"{'算法':<25} ")
            for H in H_values:
                f.write(f"H={H:<15} ")
            f.write("\n")
            f.write("-" * 80 + "\n")
            
            alg_names = ["sgd", "pipe_sgd", "localsgd", "pipe_seq_localsgd", "dream_ddp"]
            alg_display_names = ["SGD", "ASC-WFBP", "FLSGD", "PLSGD", "DreamDDP"]
            
            for alg_name, display_name in zip(alg_names, alg_display_names):
                f.write(f"{display_name:<25} ")
                for H in H_values:
                    if H in results['H_results']:
                        alg_time = results['H_results'][H]['algorithm_times'][alg_name]
                        f.write(f"{alg_time:<15.6f} ")
                    else:
                        f.write(f"{'N/A':<15} ")
                f.write("\n")
            
            f.write("="*80 + "\n\n")
            f.write("DreamDDP 速度提升总结\n")
            f.write(f"{'速度提升':<25} ")
            for H in H_values:
                f.write(f"H={H:<15} ")
            f.write("\n")
            
            speedup_names = ["over_pipe_sgd", "over_localsgd"]
            speedup_display_names = ["DreamDDP / Pipeline SGD", "DreamDDP / LocalSGD"]
            
            for speedup_name, display_name in zip(speedup_names, speedup_display_names):
                f.write(f"{display_name:<25} ")
                for H in H_values:
                    if H in results['H_results']:
                        speedup = results['H_results'][H]['speedups'][speedup_name]
                        f.write(f"{speedup:<15.4f}x ")
                    else:
                        f.write(f"{'N/A':<15} ")
                f.write("\n")
            
            f.write("="*80 + "\n")
        print(f"结果已保存到 TXT 文件: {txt_output_path}")
        
    else:
        print("\n计算失败，请检查错误信息。")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
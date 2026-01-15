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
#   train_time = time_dict[dnn][nworkers]['train']
#   comm_time = time_dict[dnn][nworkers]['comm']
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

# [gpt32_bp_sum, gpt32_comm_sum, plsgd_gpt, [wait_gpt,total_iteration_gpt], nsteps_localsgd]
def get_time_list(time_list):
    time_dict = {}
    nsteps_localsgd = time_list[4]
    time_dict["sgd"] = cal_sgd(1000, time_list[0], time_list[1])

    time_dict["pipe_sgd"] = cal_pipe_sgd(1000, time_list[0], time_list[1])

    time_dict["localsgd"] = cal_localsgd(1000, nsteps_localsgd,time_list[0], time_list[1])

    wait_list = time_list[2]
    time_dict["pipe_seq_localsgd"] = cal_pipe_seq_localsgd(1000, time_list[4],wait_list,time_list[0])

    wait_time = time_list[3][0]
    time_dict["dream_ddp"] = cal_dreamddp(1000, time_list[4], wait_time, time_list[0])
    return list(time_dict.values())
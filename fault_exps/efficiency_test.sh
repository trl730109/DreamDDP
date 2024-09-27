#!/bin/bash

hosts=('10.0.0.11' '10.0.0.12' '10.0.0.13' '10.0.0.15' '10.0.0.16' '10.0.0.18' '10.0.0.20' '10.0.0.21')
param_sync_async_op=False

# source fault_exps/resnet50_ft_baseline_sgd.sh
source fault_exps/resnet50_ft_noised_sgd_param_sync.sh  
# source fault_exps/resnet50_ft_noised_sgd_param_sync_detect.sh

param_sync_async_op=True
source fault_exps/resnet50_ft_noised_sgd_param_sync.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync_detect.sh

# hosts=('10.0.0.11' '10.0.0.12' '10.0.0.13' '10.0.0.15')
# param_sync_async_op=False

# source fault_exps/resnet50_ft_baseline_sgd.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync_detect.sh

# param_sync_async_op=True
# source fault_exps/resnet50_ft_noised_sgd_param_sync.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync_detect.sh

# hosts=('10.0.0.11' '10.0.0.12')
# param_sync_async_op=False

# source fault_exps/resnet50_ft_baseline_sgd.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync_detect.sh

# param_sync_async_op=True
# source fault_exps/resnet50_ft_noised_sgd_param_sync.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync_detect.sh

# hosts=('10.0.0.11')
# param_sync_async_op=False

# source fault_exps/resnet50_ft_baseline_sgd.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync_detect.sh

# param_sync_async_op=True
# source fault_exps/resnet50_ft_noised_sgd_param_sync.sh
# source fault_exps/resnet50_ft_noised_sgd_param_sync_detect.sh
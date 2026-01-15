#!/bin/bash
set -e  # 任意一步出错就退出（可选）

# 先跑 SGD，生成 bp
bash train_exps/transformer_sgd_lora.sh

# 再跑 LocalSGD，生成 comm
bash train_exps/transformer_localsgd_lora.sh
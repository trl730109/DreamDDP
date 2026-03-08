[![Conference](https://img.shields.io/badge/MLSys-2026-blue.svg)](https://mlsys.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **DreamDDP: Accelerating Low-Bandwidth Geo-Distributed LLM Training with Layer-wise Partial Synchronization**
> 
> *Accepted to MLSys '26*

This repository contains the official implementation and artifact materials of **DreamDDP**, a distributed training framework designed to accelerate Large Language Model (LLM) and Deep Learning (DL) model training in low-bandwidth, geo-distributed environments. 

By leveraging **layer-wise partial synchronization**, DreamDDP breaks the strict synchronization barrier of traditional Local SGD, enabling optimal overlap between backpropagation (BP) computation and parameter communication without introducing extra GPU memory overhead.

---

## 🛠️ Hardware & Software Requirements

- **Hardware**: Multi-GPU cluster with NVIDIA GPUs (e.g., RTX 2080Ti, A6000). To observe the communication bottleneck genuinely, we recommend constraining the inter-node network bandwidth (e.g., 1Gbps, 10Gbps) using Linux `tc` (Traffic Control).
- **OS**: Ubuntu 20.04 or later
- **Python**: 3.8+
- **Framework**: PyTorch 2.0+ (compiled with CUDA support)

---

## ⚙️ Installation & Setup

We highly recommend using a virtual environment (e.g., Conda) to manage dependencies.

**1. Clone the repository:**
```bash
git clone [https://github.com/trl730109/DreamDDP.git](https://github.com/trl730109/DreamDDP.git)
cd DreamDDP
```


**2. Install the required dependencies::**
```bash
# Create a conda environment (optional but recommended)
conda create -n dreamddp python=3.10 -y
conda activate dreamddp

# Install requirements
pip install -r requirements.txt
```

**3. SSH Setup:**
To enable multi-node distributed training, password-free SSH login must be configured across all machines:
* **Configure SSH Keys**: Edit the target `HOST`, `PORTS`, `USER`, and `EMAIL` variables in `ssh_conf.sh`, then run `bash ssh_conf.sh` to automatically generate and distribute the SSH keys.
* **Update Training Script**: Configure the cluster topology in `train_exps/transformer_pipeline.sh` by updating the `hosts` (e.g., `('10.244.3.188' '10.244.4.109')`), `ports` (e.g., `(22 22)`), and `master_port` variables to match your actual node IPs and SSH ports.

## 🚀 Quick Start (Profiling & Training)
```bash
bash ./train_exps/transformer_pipeline.sh
```

**4. Configure Models and Datasets:**
The models and datasets used in our paper are automatically downloaded:
* **CIFAR-10 / CIFAR-100**: Automatically downloaded via `torchvision.datasets`.
* **WikiText-2**: Manually downloaded via the HuggingFace `datasets` library for LLM (GPT-2, Llama-2, Qwen2.5-7B with LoRA) experiments.

After downloading the datasets, you should revise the data path in the configuration file, located at `./train_exps/env_configs/A6000.sh`.


## 🚀 Customizing the Training

You can customize the pipeline by editing `train_exps/transformer_pipeline.sh`:

| Option | Location | Description |
|--------|----------|-------------|
| **DNN list** | `dnn_list=(...)` | Models to run (e.g. `gpt2`, `llama2-124M`, `Qwen2.5-7B`). Add or remove entries as needed. |
| **Bandwidth** | `bandwidth="..."` | Inter-node network bandwidth (e.g. `1gbit`, `10Gbps`). Affects scheduling and profiling. |
| **DDP algorithms** | `alg='...'` blocks | Enable/disable algorithms by (un)commenting: `transformer_sgd`, `transformer_pipe_sgd`, `transformer_localsgd`, `transformer_dream_ddp`. |
| **Profile mode** | `bash transformer_pipeline.sh all` or `train` | `all` (default): profile → scheduling → training. `train`: skip profile, reuse existing data and run scheduling + training only. |

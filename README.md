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


**2. Install the required dependencies:**
```bash
conda create -n dreamddp python=3.10 -y
conda activate dreamddp
pip install -r requirements_clean.txt
```

**3. SSH Setup (multi-node only):**

Two separate networks are involved:

| Network | Used for |
|---------|----------|
| External IP + high ports (e.g. `10.249.40.11:30215`) | Running `ssh_conf.sh` from your **local machine** to push keys into the cluster |
| Internal IPs + port 22 (e.g. `10.244.x.x:22`) | Node-to-node SSH during training |

Internal nodes are network-reachable but SSH still requires key auth. `ssh_conf.sh` handles this by copying both the public key (so your local machine can SSH in) and the private key to every node (so nodes can SSH to each other via internal IPs).

Run once from your **local machine**:
```bash
# Edit HOST, PORTS, USER in ssh_conf.sh first
bash ssh_conf.sh
```

`transformer_pipeline.sh` will verify SSH connectivity (Step 0) before starting training. It does **not** configure SSH — that must be done in advance via `ssh_conf.sh`.

**4. Configure Cluster Environment:**

Edit `train_exps/env_configs/env.sh` to set paths for your cluster:
- **Python path**: run `which python3` inside your conda env to get the path, set it as `PY=...` under your `cluster_name` case.
- **Data paths**: set `data_dir` for each dataset.
- **Model paths**: set `model_dir` for each model.

**5. Download Models and Datasets:**

CIFAR-10/100 are downloaded automatically at runtime. For LLM experiments, run the provided script to download GPT-2 and WikiText-2:

```bash
# Optional: override default paths (must match env.sh)
# MODEL_DIR=/your/path DATA_DIR=/your/path bash scripts/download_gpt2_wikitext2.sh

bash scripts/download_gpt2_wikitext2.sh
```

Default paths (matching `train_exps/env_configs/env.sh`):
- GPT-2 model: `/workspace/models/gpt2`
- WikiText-2: `/workspace/wikitext2`

If you use different paths, update the corresponding `model_dir` and `data_dir` entries in `train_exps/env_configs/env.sh`.

## 🚀 Quick Start (Profiling & Training)
```bash
bash ./train_exps/transformer_pipeline.sh
```

## 🚀 Customizing the Training

You can customize the pipeline by editing `train_exps/transformer_pipeline.sh`:

| Option | Location | Description |
|--------|----------|-------------|
| **DNN list** | `dnn_list=(...)` | Models to run (e.g. `gpt2`, `llama2-124M`, `Qwen2.5-7B`). Add or remove entries as needed. |
| **Bandwidth** | `bandwidth="..."` | Inter-node network bandwidth (e.g. `1gbit`, `10Gbps`). Affects scheduling and profiling. Set `enable_tc=true` to also throttle actual traffic via `tc` (requires `cap_net_admin`). |
| **DDP algorithms** | `alg='...'` blocks | Enable/disable algorithms by (un)commenting: `transformer_sgd`, `transformer_pipe_sgd`, `transformer_localsgd`, `transformer_dream_ddp`. |
| **Profile mode** | `bash transformer_pipeline.sh all` or `train` | `all` (default): profile → scheduling → training. `train`: skip profile, reuse existing data and run scheduling + training only. |

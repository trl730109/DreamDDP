# DreamDDP 调度结果总结

**Table: Average iteration wall-clock time (in seconds) of 1000 running iterations. S₁ and S₂ represent the speedup of DreamDDP over ASC-WFBP and FLSGD.**

| Model | Qwen2.5-1.5B (8, H=5) | Qwen2.5-1.5B (32, H=5) | Qwen2.5-7B (8, H=5) | Qwen2.5-7B (32, H=5) |
|:------|:---------------------:|:----------------------:|:-------------------:|:--------------------:|
| **SGD** | 18.29 | 68.95 | 12.93 | 29.45 |
| **ASC-WFBP** | 12.49 | 42.89 | 9.20 | 19.83 |
| **FLSGD** | 6.68 | 16.77 | 5.46 | 10.18 |
| **PLSGD** | 4.89 | 14.89 | 3.97 | 9.50 |
| **DreamDDP** | **4.63** | **13.77** | **3.83** | **9.12** |
| **S₁** (vs ASC-WFBP) | 2.70× | 3.12× | 2.40× | 2.18× |
| **S₂** (vs FLSGD) | 1.44× | 1.22× | 1.42× | 1.12× |

## 详细数据

### Qwen2.5-1.5B (8 workers, H=5)
- SGD: 18.29 seconds
- ASC-WFBP: 12.49 seconds
- FLSGD: 6.68 seconds
- PLSGD: 4.89 seconds
- **DreamDDP: 4.63 seconds**
- Speedup over ASC-WFBP: **2.70×**
- Speedup over FLSGD: **1.44×**

### Qwen2.5-1.5B (32 workers, H=5)
- SGD: 68.95 seconds
- ASC-WFBP: 42.89 seconds
- FLSGD: 16.77 seconds
- PLSGD: 14.89 seconds
- **DreamDDP: 13.77 seconds**
- Speedup over ASC-WFBP: **3.12×**
- Speedup over FLSGD: **1.22×**

### Qwen2.5-7B (8 workers, H=5)
- SGD: 12.93 seconds
- ASC-WFBP: 9.20 seconds
- FLSGD: 5.46 seconds
- PLSGD: 3.97 seconds
- **DreamDDP: 3.83 seconds**
- Speedup over ASC-WFBP: **2.40×**
- Speedup over FLSGD: **1.42×**

### Qwen2.5-7B (32 workers, H=5)
- SGD: 29.45 seconds
- ASC-WFBP: 19.83 seconds
- FLSGD: 10.18 seconds
- PLSGD: 9.50 seconds
- **DreamDDP: 9.12 seconds**
- Speedup over ASC-WFBP: **2.18×**
- Speedup over FLSGD: **1.12×**

## 观察

1. **DreamDDP 在所有配置下都达到了最低的平均迭代时间**（用粗体标记）。
2. **Qwen2.5-1.5B** 在 8 和 32 workers 下（H=5），DreamDDP 相比 ASC-WFBP 的加速比分别为 **2.70×** 和 **3.12×**，相比 FLSGD 的加速比分别为 **1.44×** 和 **1.22×**，显示出显著的性能提升。
3. **Qwen2.5-7B** 在 8 和 32 workers 下（H=5），DreamDDP 相比 ASC-WFBP 的加速比分别为 **2.40×** 和 **2.18×**，相比 FLSGD 的加速比分别为 **1.42×** 和 **1.12×**。
4. 随着 worker 数量增加，DreamDDP 相比 ASC-WFBP 的加速比在 Qwen2.5-1.5B 上增加（2.70× → 3.12×），但在 Qwen2.5-7B 上略有下降（2.40× → 2.18×）。
5. 所有配置现在都使用 H=5，DreamDDP 在所有配置下都保持最佳性能，相比 ASC-WFBP 的加速比在 2.18× 到 3.12× 之间。


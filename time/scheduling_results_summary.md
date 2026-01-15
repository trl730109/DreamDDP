# DreamDDP 调度结果总结

**Table: Average iteration wall-clock time (in seconds) of 1000 running iterations. S₁ and S₂ represent the speedup of DreamDDP over Pipeline SGD and LocalSGD.**

| Model | Qwen2.5-1.5B (8) | Qwen2.5-1.5B (32) | Qwen2.5-7B (8) | Qwen2.5-7B (32) |
|:------|:----------------:|:-----------------:|:--------------:|:---------------:|
| **SGD** | 26.05 | 164.78 | 4.54 | 15.63 |
| **Pipeline SGD** | 16.39 | 99.63 | 4.16 | 10.81 |
| **LocalSGD** | 4.29 | 18.03 | 3.70 | 4.79 |
| **Pipeline Seq LocalSGD** | 3.45 | 17.32 | 3.61 | 4.38 |
| **DreamDDP** | **3.12** | **16.51** | **3.61** | **4.21** |
| **S₁** (vs Pipeline SGD) | 5.25× | 6.03× | 1.15× | 2.57× |
| **S₂** (vs LocalSGD) | 1.38× | 1.09× | 1.02× | 1.14× |

## 详细数据

### Qwen2.5-1.5B (8 workers, H=10)
- SGD: 26.05 seconds
- Pipeline SGD: 16.39 seconds
- LocalSGD: 4.29 seconds
- Pipeline Seq LocalSGD: 3.45 seconds
- **DreamDDP: 3.12 seconds**
- Speedup over Pipeline SGD: **5.25×**
- Speedup over LocalSGD: **1.38×**

### Qwen2.5-1.5B (32 workers, H=10)
- SGD: 164.78 seconds
- Pipeline SGD: 99.63 seconds
- LocalSGD: 18.03 seconds
- Pipeline Seq LocalSGD: 17.32 seconds
- **DreamDDP: 16.51 seconds**
- Speedup over Pipeline SGD: **6.03×**
- Speedup over LocalSGD: **1.09×**

### Qwen2.5-7B (8 workers, H=10)
- SGD: 4.54 seconds
- Pipeline SGD: 4.16 seconds
- LocalSGD: 3.70 seconds
- Pipeline Seq LocalSGD: 3.61 seconds
- **DreamDDP: 3.61 seconds**
- Speedup over Pipeline SGD: **1.15×**
- Speedup over LocalSGD: **1.02×**

### Qwen2.5-7B (32 workers, H=10)
- SGD: 15.63 seconds
- Pipeline SGD: 10.81 seconds
- LocalSGD: 4.79 seconds
- Pipeline Seq LocalSGD: 4.38 seconds
- **DreamDDP: 4.21 seconds**
- Speedup over Pipeline SGD: **2.57×**
- Speedup over LocalSGD: **1.14×**

## 观察

1. **DreamDDP 在所有配置下都达到了最低的平均迭代时间**（用粗体标记）。
2. **Qwen2.5-1.5B** 在 8 和 32 workers 下，DreamDDP 相比 Pipeline SGD 的加速比分别为 **5.25×** 和 **6.03×**，显示出显著的性能提升。
3. **Qwen2.5-7B** 在 8 workers 下，DreamDDP 与 Pipeline Seq LocalSGD 的性能非常接近（3.61 vs 3.61），但在 32 workers 下仍保持优势。
4. 随着 worker 数量增加，DreamDDP 相比 Pipeline SGD 的加速比通常会增加（Qwen2.5-1.5B: 5.25× → 6.03×，Qwen2.5-7B: 1.15× → 2.57×）。


import torch
import time

# 确保使用的是 Ampere 架构显卡 (如 A6000)
print(f"Device: {torch.cuda.get_device_name(0)}")

# 初始化大矩阵
N = 8192
A = torch.randn(N, N, device='cuda')
B = torch.randn(N, N, device='cuda')

# GPU预热 (必不可少，防止冷启动误差)
_ = A @ B
torch.cuda.synchronize()

# 1. 测试 TF32 开启时的时间 (Ampere 默认行为)
torch.backends.cuda.matmul.allow_tf32 = True
t0 = time.time()
for _ in range(10):
    _ = A @ B
torch.cuda.synchronize()
print(f"[TF32 ON]  耗时: {time.time() - t0:.4f} 秒")

# 2. 测试 TF32 关闭时的时间 (你要模拟的状态)
torch.backends.cuda.matmul.allow_tf32 = False
t0 = time.time()
for _ in range(10):
    _ = A @ B
torch.cuda.synchronize()
print(f"[TF32 OFF] 耗时: {time.time() - t0:.4f} 秒")
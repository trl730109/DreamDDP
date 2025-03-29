import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # 第一层：2 输入，3 输出
        self.fc2 = nn.Linear(3, 1)  # 第二层：3 输入，1 输出
    
    def forward(self, x):
        y = self.fc1(x)
        z = self.fc2(y)
        return z

# 创建模型实例
net = SimpleNet()

# 创建 SGD 优化器
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 模拟 trainer 对象（通常 trainer 是一个封装类）
class Trainer:
    def __init__(self, net, optimizer):
        self.net = net
        self.optimizer = optimizer


trainer = Trainer(net, optimizer)
print("=== trainer.net.named_parameters ===")
for name, parameter in trainer.net.named_parameters():
    print(name)
    print(parameter)

print("=== trainer.net.parameters ===")
for i, param in enumerate(trainer.net.parameters()):
    print(f"Parameter {i}:")
    print(param.grad.data)

# 打印 param_groups
print("=== trainer.optimizer.param_groups ===")
for i, group in enumerate(trainer.optimizer.param_groups):
    print(f"Parameter Group {i}:")
    print(group)  # 直接打印整个字典

# 打印 group['params']
print("\n=== group['params'] ===")
for i, group in enumerate(trainer.optimizer.param_groups):
    for j, p in enumerate(group['params']):
        print(f"  Param {j}: shape={p.shape}, requires_grad={p.requires_grad}")

# 运行一步优化，生成 momentum_buffer（可选）
optimizer.zero_grad()
input = torch.randn(1, 2)
output = net(input)
loss = output.sum()
loss.backward()
optimizer.step()

print("\n=== optimizer.state_dict() ===")
print(trainer.optimizer.state_dict())
print("\n=== optimizer.state ===")
print(trainer.optimizer.state)


import torch.nn as nn

# 定义一个简单的网络
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)  # 叶子模块
        self.fc = nn.Sequential(         # 非叶子模块
            nn.Linear(16, 10),           # 叶子模块
            nn.ReLU()                    # 叶子模块
        )

net = MyNet()

# 遍历模块
for name, module in net.named_modules():
    print(f"Name: {name}, Is Leaf: {len(list(module.children())) == 0}")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import json

# Data loading
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainset = datasets.CIFAR100(root='/data2/share/zhtang/cifar100', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Model setup
model = models.resnet50(pretrained=False, num_classes=100)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# Training loop
def train(model, trainloader, criterion, optimizer, iterations):
    model.train()
    grad_magnitudes = {10: {}, 50: {}, 100: {}}
    for i, (inputs, targets) in enumerate(trainloader):
        if i > iterations:
            break
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Obtain gradients at specified iterations
        if i in [10, 50, 100]:
            for name, param in model.named_parameters():
                if 'layer1.0.conv1.weight' in name or 'layer2.0.conv1.weight' in name:
                    grad_magnitudes[i][name] = param.grad.data.clone()

        optimizer.step()

    # Save gradients to a file
    torch.save(grad_magnitudes, 'resnet50_gradients.pth')

train(model, trainloader, criterion, optimizer, iterations=100)















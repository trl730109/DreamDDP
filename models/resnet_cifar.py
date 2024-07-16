from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


from .group_normalization import GroupNorm2d
from .basics import View


def norm2d(planes, num_channels_per_group=32):
    print("num_channels_per_group:{}".format(num_channels_per_group))
    if num_channels_per_group > 0:
        return GroupNorm2d(
            planes, num_channels_per_group, affine=True, track_running_stats=False
        )
    else:
        return nn.BatchNorm2d(planes)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = norm2d(planes, group_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, group_norm=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = norm2d(planes, group_norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = norm2d(planes, group_norm)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.bn3 = norm2d(planes * self.expansion, group_norm)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3):
        super(CifarResNet, self).__init__()
        # self.in_planes = 64
        self.res_base_width = res_base_width
        self.in_planes = self.res_base_width

        self.conv1 = nn.Conv2d(in_channels, self.res_base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm2d(self.res_base_width, group_norm)
        self.layer1 = self._make_layer(block, self.res_base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.res_base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.res_base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.res_base_width*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(self.res_base_width * 8 * block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)

        if return_features:
            return out, feature
        else:
            return out



def get_res18_out_channels(res_base_width):
    out_channels = []
    num_blocks = [2, 2, 2, 2]
    in_planes = res_base_width

    def _make_layer(planes, num_blocks, stride):
        nonlocal in_planes
        nonlocal out_channels
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            out_channels.append(planes)
            in_planes = planes
    out_channels.append(res_base_width)
    _make_layer(res_base_width, num_blocks[0], stride=1)
    _make_layer(res_base_width*2, num_blocks[1], stride=2)
    _make_layer(res_base_width*4, num_blocks[2], stride=2)
    _make_layer(res_base_width*8, num_blocks[3], stride=2)
    return out_channels




def make_ResNet_seqs(init_classifier, 
            block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3):
    # in_planes = 64
    in_planes = res_base_width
    # layers = nn.ModuleList([])
    layers = []

    def _make_layer(block, planes, num_blocks, stride):
        nonlocal in_planes
        nonlocal layers
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes * block.expansion

    layers.append(
        nn.Sequential(nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
                    norm2d(res_base_width, group_norm),
                    nn.ReLU())
            )

    _make_layer(block, res_base_width, num_blocks[0], stride=1)
    _make_layer(block, res_base_width*2, num_blocks[1], stride=2)
    _make_layer(block, res_base_width*4, num_blocks[2], stride=2)
    _make_layer(block, res_base_width*8, num_blocks[3], stride=2)
    # linear = nn.Linear(res_base_width * 8 * block.expansion, num_classes)
    # torch.nn.AvgPool2d(kernel_size, stride=None, padding=0)
    layers.append(
        nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                        View([-1]))
                    )
    layers.append(nn.Linear(res_base_width * 8 * block.expansion, num_classes))
    return layers


def make_ResNet_Head_seqs(init_classifier, split_layer_index,  
            block, num_blocks, num_classes=10, group_norm=0, res_base_width=64, in_channels=3):
    # in_planes = 64
    # args = args
    origin_res_layer_index = 0
    in_planes = res_base_width
    layers = []

    def _make_layer(block, planes, num_blocks, stride):
        nonlocal in_planes
        nonlocal layers
        nonlocal origin_res_layer_index
        nonlocal split_layer_index
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            if origin_res_layer_index > split_layer_index:
                layers.append(block(in_planes, planes, stride))
                in_planes = planes * block.expansion
            origin_res_layer_index += 1

    if origin_res_layer_index > split_layer_index:
        layers.append(
            nn.Sequential(nn.Conv2d(in_channels, res_base_width, kernel_size=3, stride=1, padding=1, bias=False),
                        norm2d(res_base_width, group_norm),
                        nn.ReLU())
                )
    origin_res_layer_index += 1

    _make_layer(block, res_base_width, num_blocks[0], stride=1)
    _make_layer(block, res_base_width*2, num_blocks[1], stride=2)
    _make_layer(block, res_base_width*4, num_blocks[2], stride=2)
    _make_layer(block, res_base_width*8, num_blocks[3], stride=2)
    if origin_res_layer_index > split_layer_index:
        layers.append(
            nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                            View([-1]))
                        )
    origin_res_layer_index += 1

    if init_classifier:
        if origin_res_layer_index > split_layer_index:
            layers.append(nn.Linear(res_base_width * 8 * block.expansion, num_classes))
        origin_res_layer_index += 1

    return layers





def resnet18_layers(init_classifier, num_classes=10, **kwargs):
    return make_ResNet_seqs(init_classifier, BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


def resnet34_layers(init_classifier, num_classes=10, **kwargs):
    return make_ResNet_seqs(init_classifier, BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)

def resnet50_layers(init_classifier, num_classes=10, **kwargs):
    return make_ResNet_seqs(init_classifier, Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def resnet18_head(init_classifier, split_layer_index, **kwargs):
    return make_ResNet_Head_seqs(init_classifier, split_layer_index, BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50_head(init_classifier, split_layer_index, **kwargs):
    return make_ResNet_Head_seqs(init_classifier, split_layer_index, Bottleneck, [3, 4, 6, 3], **kwargs)





def cifar_resnet18(num_classes=10, **kwargs):
    return CifarResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


def cifar_resnet34(num_classes=10, **kwargs):
    return CifarResNet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


def cifar_resnet50(num_classes=10, **kwargs):
    return CifarResNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def cifar_resnet101(num_classes=10, **kwargs):
    return CifarResNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def cifar_resnet152(num_classes=10, **kwargs):
    return CifarResNet(Bottleneck, [3, 8, 36, 3], num_classes, **kwargs)












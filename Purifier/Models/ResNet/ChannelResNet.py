import torch
import torch.nn as nn
import sys
sys.path.append('/home/jinyulin/Purifier/')
from torch.nn import functional as F
from Purifier.config import config

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Channel_Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Channel_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if config['Global']['dataset'] == 'GTSRB':
            self.fc = nn.Linear(planes, 43)
        else:
            self.fc = nn.Linear(planes, 10)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x, label=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        fc_out = self.fc(torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=-1))


        if self.training:
            N, C, _, _ = out.shape
            mask = self.fc.weight[label, :]
            topk = torch.topk(fc_out, k=2, dim=1)[1][:,1]
            mask1 = self.fc.weight[topk, :]
            out = out/2 * (mask+mask1).view(N, C, 1, 1)
            out = out * (mask).view(N, C, 1, 1)
        else:
            N, C, _, _ = out.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            topk = torch.topk(fc_out, k=2, dim=1)[1][:,1]
            mask = self.fc.weight[pred_label, :]
            mask1 = self.fc.weight[topk, :]
            out = out * (mask).view(N, C, 1, 1)
            out = out/2 * (mask+mask1).view(N, C, 1, 1)
        return out, fc_out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Channel_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Channel_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_(Channel_Block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)

    def forward(self, x, y=None, eval=True):
        if eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()
        
        class_wise_output = list()

        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.layer3(out)
        
        for layer in self.layer4:
            out, layer2_out = layer(out, y)
            class_wise_output.append(layer2_out)
        
        
        
        out1 = F.avg_pool2d(out, 4)
        out2 = out1.view(out1.size(0), -1)
        out1 = self.fc(out2)
        return {
            'normal':out,
            'auxiliary':class_wise_output
        }


def channel_resnet18(**kwargs):
    return Channel_ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

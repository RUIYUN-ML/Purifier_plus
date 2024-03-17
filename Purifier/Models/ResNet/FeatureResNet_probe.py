import sys
sys.path.append('/home/jinyulin/Purifier/')
import torch
import torch.nn as nn
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

# ------------
class Global_Avg_Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        return out
    

class nlBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(nlBlock, self).__init__()
        self.global_avg_flatten = Global_Avg_Flatten()
        self.fc = nn.Linear(in_planes, planes)
        self.softplus = nn.Softplus(beta=5)

    def forward(self, x):
        out = self.softplus(self.fc(self.global_avg_flatten(x)))
        return out
    

class CIFS(nn.Module):
    def __init__(self, n_feat=512, n_cls = 10, mode='linear'):
        super().__init__()
        if mode == 'linear':
            self.Probe = nn.Sequential(*[Global_Avg_Flatten(), nn.Linear(n_feat, n_cls)])
        else:
            self.Probe = nn.Sequential(*[nlBlock(n_feat, 128), nn.Linear(128, n_cls)])

    def forward(self, feat, y=None):
        ''' # y: (batch), feat: (batch, 512, h, w); ## masked feat: (batch, n_cls), cas prediction: (batch, 512) '''
        Mask = self._get_mask_with_graph(feat, y)
        pred_raw = self.Probe(feat)  # (batch, n_cls)
        masked_feat = feat * Mask
        return masked_feat, pred_raw

    def _get_mask_with_graph(self, feat, y=None):
        N, C, H, W = feat.shape
        feat = feat.detach().clone()
        feat.requires_grad_(True)

        logits = self.Probe(feat)  # (batch, 10)
        if not self.training:
            pred = logits.topk(k=2, dim=1)[1]
            pred_t1 = pred[:, 0]
            pred_t2 = pred[:, 1]
            top1_logit = logits[torch.tensor(list(range(N))), pred_t1].sum()
            top2_logit = logits[torch.tensor(list(range(N))), pred_t2].sum()
        else:
            pred = logits.topk(k=2, dim=1)[1]
            pred_t2 = pred[:, 1]
            top1_logit = logits[torch.tensor(list(range(N))), y].sum()
            top2_logit = logits[torch.tensor(list(range(N))), pred_t2].sum()

        max_logit = top1_logit + top2_logit
        mask = torch.autograd.grad(max_logit, feat, create_graph=True)[0]

        # 下面就是CIFS主要修改的地方
        mask = mask.view(mask.shape[0], -1)
        mask = F.softmax(mask, dim=1)
        return mask.view(N, C, H, W)
    
    def _requires_grad(self, requires_grad):
        for param in self.parameters():
            param.requires_grad = requires_grad


class Feature_Probe_Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, size, stride=1, mode='linear'):
        super(Feature_Probe_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        # CIFS
        if config['Global']['dataset'] == 'GTSRB':
            self.Probe = CIFS(planes, 43, mode)
        else: 
            self.Probe = CIFS(planes, 10, mode)
        # Feature_Purifier
        if config['Global']['dataset'] == 'GTSRB':
            self.fc = nn.Linear(size*planes, 43)
        else:
            self.fc = nn.Linear(size*planes, 10)

    def forward(self, x, label=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        # Feature_Purifier
        fc_out = self.fc(out.view(out.shape[0],-1))
        if self.training:
            N, C, H, W = out.shape
            mask = self.fc.weight[label, :]
            out = out * mask.view(N, C, H, W)
        else:
            N, C, H, W = out.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            mask = self.fc.weight[pred_label, :]
            out = out * mask.view(N, C, H, W)
        #CIFS
        masked_out, pred_raw = self.Probe(out, label)

        return masked_out, pred_raw, out, fc_out


class Feature_Probe_Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Feature_Probe_Resnet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_(Feature_Probe_Block, 512, num_blocks[3], stride=2, size = 4*4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _make_layer_(self, block, planes, num_blocks, stride, size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, size, stride))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(layers)
    
    def forward(self, x, y=None, eval=True):
        if eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()

        pred_raw_list = list()
        class_wise_output = list()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        for layer in self.layer4:
            out, pred_raw, _, layer4_out = layer(out, y)
            pred_raw_list.append(pred_raw)
            class_wise_output.append(layer4_out)
        

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return {
            'normal': out,
            'CIFS_pred_raw': pred_raw_list,  
            'auxiliary': class_wise_output
        }


def feature_probe_resnet18(**kwargs):
    return Feature_Probe_Resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
# ------------
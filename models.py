import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, args):
        super(SimpleCNN, self).__init__()
        self.args = args
        self.conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv0_bn = nn.BatchNorm2d(64)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(256, args.num_classes)

    def forward(self, x):
        out = x
        out = F.leaky_relu_(self.conv0_bn(self.conv0(out)))
        out = F.max_pool2d(out, 2)
        out = F.leaky_relu_(self.conv1_bn(self.conv1(out)))
        out = F.max_pool2d(out, 2)
        out = F.leaky_relu_(self.conv2_bn(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = F.leaky_relu_(self.conv3_bn(self.conv3(out)))
        out = F.max_pool2d(out, 2)
        if out.shape[-1] != 2:
            out = F.adaptive_avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
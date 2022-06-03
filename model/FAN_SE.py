import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes:int, out_planes:int, stride=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=padding, bias=bias)

def conv1x1(in_planes:int, out_planes:int, bias=True):
    "1x1 convolution without padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     stride=1, padding=0, bias=bias)

class SELayer_old(nn.Module):
    def __init__(self, in_planes:int, reduction=4):
        super(SELayer, self).__init__()
        self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                conv1x1(in_planes, in_planes // reduction, bias=False),
                # nn.BatchNorm2d(in_planes // reduction),
                nn.ReLU(inplace=True),
                conv1x1(in_planes // reduction, in_planes, bias=False),
                # nn.BatchNorm2d(in_planes),
                nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_planes:int, out_planes:int):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        self.relu = nn.ReLU(inplace=True)
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = self.relu(out3)
        out3 = self.conv3(out3)

        out3 = torch.cat([out1, out2, out3], axis=1)
        if self.shortcut != None:
            residual =  self.shortcut(residual)
        out3 += residual

        return out3

class HourGlassNet(nn.Module):
    def __init__(self, depth:int, num_feats:int):
        super(HourGlassNet, self).__init__()
        self.depth = depth
        self.num_feats = num_feats

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        for level in range(1, depth + 1):
            # upper branch
            self.add_module(f"shortcut{level}", ConvBlock(self.num_feats, self.num_feats))
            # lower branch
            self.add_module(f"conv{level}_1", ConvBlock(self.num_feats, self.num_feats))
            self.add_module(f"conv{level}_2", ConvBlock(self.num_feats, self.num_feats))

            self.add_module(f"SE{level}", SELayer(self.num_feats))

            if level == depth:
                self.add_module(f"conv_middle", ConvBlock(self.num_feats, self.num_feats))

    def _forward(self, x, level:int):
        residual = x
        # upper branch
        residual = self._modules[f"shortcut{level}"](residual)
        # lower branch
        x = self.downsample(x)
        x = self._modules[f"conv{level}_1"](x)
        if level == self.depth:
            # End recursion
            x = self._modules["conv_middle"](x)
        else:
            # Recursion forward
            x = self._forward(x, level + 1)
            
        
        x = self._modules[f"conv{level}_2"](x)
        x = self._modules[f"SE{level}"](x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        return x + residual

    def forward(self, x):
        return self._forward(x, 1)

class FAN(nn.Module):
    """Facial Alignment network
    """
    def __init__(self, num_HG:int = 4, HG_depth:int = 4, num_feats:int = 256, num_classes:int = 68):
        super(FAN, self).__init__()
        self.num_HG = num_HG # num of how many hourglass stack
        self.HG_dpeth = HG_depth # num of recursion in hourglass net
        self.num_feats = num_feats
        self.num_classes = num_classes # num of keypoints

        # Base part
        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(3, int(self.num_feats / 4), kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(int(self.num_feats / 4))
        self.conv2 = ConvBlock(int(self.num_feats / 4), int(self.num_feats / 2))
        self.conv3 = ConvBlock(int(self.num_feats / 2), int(self.num_feats / 2))
        self.conv4 = ConvBlock(int(self.num_feats / 2), self.num_feats)

        # Stacked hourglassNet part
        for stack_idx in range(1, self.num_HG + 1):
            self.add_module(f"HG{stack_idx}", HourGlassNet(self.HG_dpeth, self.num_feats))
            self.add_module(f"stack{stack_idx}_conv1", ConvBlock(self.num_feats, self.num_feats))
            self.add_module(f"stack{stack_idx}_conv2", conv1x1(self.num_feats, self.num_feats))
            self.add_module(f"stack{stack_idx}_bn1", nn.BatchNorm2d(int(self.num_feats)))

            self.add_module(f"stack{stack_idx}_conv_out", conv1x1(self.num_feats, self.num_classes))
            if stack_idx != self.num_HG:
                self.add_module(f"stack{stack_idx}_conv3", conv1x1(self.num_feats, self.num_feats))
                self.add_module(f"stack{stack_idx}_shortcut", conv1x1(self.num_classes, self.num_feats))
        self._weight_init()
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        # Base part
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Stacked hourglassNet part
        for stack_idx in range(1, self.num_HG + 1):
            if stack_idx != self.num_HG:
                residual = x
            x = self._modules[f"HG{stack_idx}"](x)
            x = self._modules[f"stack{stack_idx}_conv1"](x)
            x = self._modules[f"stack{stack_idx}_conv2"](x)
            x = self.relu(self._modules[f"stack{stack_idx}_bn1"](x))
            # Output heatmap
            out = self._modules[f"stack{stack_idx}_conv_out"](x)
            outputs.append(out)
            # lower and upper branch
            if stack_idx != self.num_HG:
                x = self._modules[f"stack{stack_idx}_conv3"](x)
                out_ = self._modules[f"stack{stack_idx}_shortcut"](out)
                x = out_ + residual + x
        return outputs
            



    


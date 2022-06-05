import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(inplanes:int, planes:int, stride=1, padding=1, bias=False, dilation=1):
    "3x3 convolution"
    inplanes = int(inplanes)
    planes = int(planes)
    return nn.Conv2d(inplanes, planes, kernel_size=3, dilation=dilation,
                     stride=stride, padding=padding, bias=bias)

def conv1x1(inplanes:int, planes:int, bias=False):
    "1x1 convolution"
    inplanes = int(inplanes)
    planes = int(planes)
    return nn.Conv2d(inplanes, planes, kernel_size=1,bias=bias,
                     stride=1, padding=0)

def depthwise_conv3x3(planes:int, stride=1, padding=1, bias=False, dilation=1):
    "3x3 depthwise convolution "
    return nn.Conv2d(planes, planes, kernel_size=3, dilation=dilation,
                     stride=stride, padding=padding, bias=bias, groups=planes)

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

class CA_Block(nn.Module):
    """Coordinate Attention Block
    """
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):

        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1):
        super(InvertedResidual, self).__init__()

        self.identity = stride == 1 and inp == oup
        hidden_dim = inp * 2
      
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # Squeeze-and-Excite
            SELayer(hidden_dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Bottleneck(nn.Module):
    def __init__(self, inplanes:int, planes:int):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes // 2)
        self.bn1 = nn.BatchNorm2d(planes // 2)
        self.conv2 = conv3x3(planes // 2, planes // 2)
        self.bn2 = nn.BatchNorm2d(planes // 2)
        self.conv3 = conv1x1(planes // 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if inplanes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(inplanes, planes),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut != None:
            residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)

        return out

class HPM_ConvBlock(nn.Module):
    """Hierarchical, parallel and multi-scale block
    """
    def __init__(self, inplanes:int, planes:int):
        super(HPM_ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes // 2)
        self.bn2 = nn.BatchNorm2d(planes // 2)
        self.conv2 = conv3x3(planes // 2, planes // 4)
        self.bn3 = nn.BatchNorm2d(planes // 4)
        self.conv3 = conv3x3(planes // 4, planes // 4)

        self.relu = nn.ReLU(inplace=True)
        if inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(inplace=True),
                conv1x1(inplanes, planes)
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
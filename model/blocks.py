from turtle import forward
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
    def __init__(self, inp, oup, reduction=4):
        super(CA_Block, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

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

class HPM_ConvBlock_gn(nn.Module):
    """Hierarchical, parallel and multi-scale block
    """
    def __init__(self, inplanes:int, planes:int):
        super(HPM_ConvBlock, self).__init__()
        self.bn1 = nn.GroupNorm(32,inplanes)
        self.conv1 = conv3x3(inplanes, planes // 2)
        self.bn2 = nn.GroupNorm(32,inplanes)
        self.conv2 = conv3x3(planes // 2, planes // 4)
        self.bn3 = nn.GroupNorm(32,inplanes)
        self.conv3 = conv3x3(planes // 4, planes // 4)

        self.relu = nn.ReLU(inplace=True)
        if inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.GroupNorm(32,inplanes),
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

class HPM_ConvBlock_SD(nn.Module):
    """Hierarchical, parallel and multi-scale block
    """
    def __init__(self, inplanes:int, planes:int, prob=1.0):
        super(HPM_ConvBlock_SD, self).__init__()
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

        self.prob = torch.tensor(prob)
        self.start_drop = False
    def _forward(self, x):
        
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
    def forward(self, x):
        if self.training and self.start_drop:
            active = torch.bernoulli(self.prob)
            if active == 1:
                return self._forward(x)
            else:
                if self.shortcut != None:
                    return self.shortcut(x)
                else:
                    return x
        else:
            return self._forward(x)
        
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

        self.xx_channel = None
        self.yy_channel = None
        self.r_channel = None
        self.speed_up = True

    def get_xxyy(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()
        def gen_xx_yy():
            xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
            yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

            xx_channel = xx_channel / (x_dim - 1)
            yy_channel = yy_channel / (y_dim - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
            yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
            
            return xx_channel, yy_channel

        if self.speed_up:
            if self.xx_channel == None:
                self.xx_channel, self.yy_channel = gen_xx_yy()
                if input_tensor.is_cuda:
                    self.xx_channel = self.xx_channel.cuda()
                    self.yy_channel = self.yy_channel.cuda()
            return self.xx_channel[:batch_size].clone(), self.yy_channel[:batch_size].clone()
        else:
            xx_channel, yy_channel = gen_xx_yy()
            if input_tensor.is_cuda:
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda() 
            return xx_channel, yy_channel
            
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        
        xx_channel, yy_channel = self.get_xxyy(input_tensor)
        ret = torch.cat([
                    input_tensor,
                    xx_channel.type_as(input_tensor),
                    yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = self.get_r(input_tensor.size())
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            if input_tensor.is_cuda:
                rr = rr.cuda()
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        extra_channel = 3 if with_r else 2
        self.conv = nn.Conv2d(in_channels + extra_channel, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret

class AddCoordsTh(nn.Module):
    def __init__(self, x_dim=96, y_dim=96, with_r=False, with_boundary=False):
        super(AddCoordsTh, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r
        self.with_boundary = with_boundary

    def forward(self, input_tensor, heatmap=None):
        """
        input_tensor: (batch, c, x_dim, y_dim)
        """
        batch_size_tensor = input_tensor.shape[0]

        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32).cuda()
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0).cuda()
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones.float(), xx_range.float())
        xx_channel = xx_channel.unsqueeze(-1)


        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32).cuda()
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0).cuda()
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range.float(), yy_ones.float())
        yy_channel = yy_channel.unsqueeze(-1)

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)

        xx_channel = xx_channel / (self.x_dim - 1)
        yy_channel = yy_channel / (self.y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        if self.with_boundary and type(heatmap) != type(None):
            boundary_channel = torch.clamp(heatmap[:, -1:, :, :],
                                        0.0, 1.0)

            zero_tensor = torch.zeros_like(xx_channel)
            xx_boundary_channel = torch.where(boundary_channel>0.05,
                                              xx_channel, zero_tensor)
            yy_boundary_channel = torch.where(boundary_channel>0.05,
                                              yy_channel, zero_tensor)
        if self.with_boundary and type(heatmap) != type(None):
            xx_boundary_channel = xx_boundary_channel.cuda()
            yy_boundary_channel = yy_boundary_channel.cuda()
        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)


        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            rr = rr / torch.max(rr)
            ret = torch.cat([ret, rr], dim=1)

        if self.with_boundary and type(heatmap) != type(None):
            ret = torch.cat([ret, xx_boundary_channel,
                             yy_boundary_channel], dim=1)
        return ret


class CoordConvTh(nn.Module):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, with_boundary,
                 in_channels, first_one=False, *args, **kwargs):
        super(CoordConvTh, self).__init__()
        self.addcoords = AddCoordsTh(x_dim=x_dim, y_dim=y_dim, with_r=with_r,
                                    with_boundary=with_boundary)
        in_channels += 2
        if with_r:
            in_channels += 1
        if with_boundary and not first_one:
            in_channels += 2
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    def forward(self, input_tensor, heatmap=None):
        ret = self.addcoords(input_tensor, heatmap)
        last_channel = ret[:, -2:, :, :]
        ret = self.conv(ret)
        return ret, last_channel
import torch
import torch.nn as nn

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
        
        b,c,h,w = x.size()
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

        # self.attention = CA_Block(planes, planes)
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
        # out3 = self.attention(out3)
        if self.shortcut != None:
            residual =  self.shortcut(residual)
        out3 += residual

        return out3
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
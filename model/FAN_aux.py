import torch
import torch.nn as nn
import torch.nn.functional as F
from model.blocks import conv1x1, conv3x3
from model.blocks import HPM_ConvBlock, SELayer, CA_Block
from model.blocks import CoordConv
import math

class AuxiliaryNet(nn.Module):
    def __init__(self, input_channels, nums_class=3, activation=nn.ReLU):
        super(AuxiliaryNet, self).__init__()
        self.input_channels = input_channels
        self.num_channels = [256, 256, 64, 256, 64]
        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels[0])

        self.conv2 = nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels[1])

        self.conv3 = nn.Conv2d(self.num_channels[1], self.num_channels[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels[2])

        self.conv4 = nn.Conv2d(self.num_channels[2], self.num_channels[3], kernel_size=7, stride=1, padding=3)
        self.bn4 = nn.BatchNorm2d(self.num_channels[3])

        self.fc1 = nn.Linear(in_features=self.num_channels[3], out_features=self.num_channels[4])
        self.fc2 = nn.Linear(in_features=self.num_channels[4], out_features=nums_class)

        self.activation = activation(inplace=True)

        self.init_params()

    def init_params(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #print(m.weight.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.activation(out)

        out = F.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        #print(out.size())
        # out = out.view(out.size(0), -1)
        out = self.fc1(out)
        euler_angles_pre = self.fc2(out)

        return euler_angles_pre

class HourGlassNet(nn.Module):
    def __init__(self, depth:int, num_feats:int, resBlock=HPM_ConvBlock, attention_block=None, add_CoordConv=False, with_r=False):
        super(HourGlassNet, self).__init__()
        self.depth = depth
        self.num_feats = num_feats

        self.add_CoordConv = add_CoordConv
        if self.add_CoordConv:
            self.coordConv = CoordConv(self.num_feats, self.num_feats, with_r=with_r, kernel_size=1, stride=1, padding=0)
        self.attention_block = attention_block
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        for level in range(1, depth + 1):
            # upper branch
            self.add_module(f"shortcut{level}", resBlock(self.num_feats, self.num_feats))
            # lower branch
            self.add_module(f"conv{level}_1", resBlock(self.num_feats, self.num_feats))
            self.add_module(f"conv{level}_2", resBlock(self.num_feats, self.num_feats))

            if attention_block != None:
                if attention_block == SELayer:
                    self.add_module(f"attention{level}", attention_block(self.num_feats))
                elif attention_block == CA_Block:
                    self.add_module(f"attention{level}", attention_block(self.num_feats, self.num_feats))
                else:
                    raise ValueError("This attention block doesn't exist!")
            if level == depth:
                self.add_module(f"conv_middle", resBlock(self.num_feats, self.num_feats))

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
        if self.attention_block != None:
            x = self._modules[f"attention{level}"](x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        return x + residual

    def forward(self, x):
        if self.add_CoordConv:
            x = self.coordConv(x)

        return self._forward(x, 1)

class FAN_aux(nn.Module):
    """Facial Alignment network
    """
    def __init__(self, num_HG:int = 4, HG_depth:int = 4, num_feats:int = 256, 
                num_classes:int = 68, resBlock=HPM_ConvBlock, attention_block=None,use_CoordConv=False, with_r=False, add_CoordConv_inHG=False):
        super(FAN_aux, self).__init__()
        self.num_HG = num_HG # num of how many hourglass stack
        self.HG_dpeth = HG_depth # num of recursion in hourglass net
        self.num_feats = num_feats
        self.num_classes = num_classes # num of keypoints

        # Base part
        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_CoordConv:
            self.conv1 = CoordConv(3, self.num_feats // 4, with_r=with_r, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(3, self.num_feats // 4, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.num_feats // 4)
        self.conv2 = resBlock(self.num_feats // 4, self.num_feats // 2)
        self.conv3 = resBlock(self.num_feats // 2, self.num_feats // 2)
        self.conv4 = resBlock(self.num_feats // 2, self.num_feats)

        # Stacked hourglassNet part
        for stack_idx in range(1, self.num_HG + 1):
            self.add_module(f"HG{stack_idx}", HourGlassNet(self.HG_dpeth, self.num_feats, resBlock=resBlock, attention_block=attention_block, add_CoordConv=add_CoordConv_inHG, with_r=with_r))
            self.add_module(f"stack{stack_idx}_conv1", resBlock(self.num_feats, self.num_feats))
            self.add_module(f"stack{stack_idx}_conv2", conv1x1(self.num_feats, self.num_feats, bias=True))
            self.add_module(f"stack{stack_idx}_bn1", nn.BatchNorm2d(int(self.num_feats)))

            self.add_module(f"stack{stack_idx}_conv_out", conv1x1(self.num_feats, self.num_classes, bias=True))
            if stack_idx != self.num_HG:
                self.add_module(f"stack{stack_idx}_conv3", conv1x1(self.num_feats, self.num_feats, bias=True))
                self.add_module(f"stack{stack_idx}_shortcut", conv1x1(self.num_classes, self.num_feats, bias=True))
        self.aux_net = AuxiliaryNet(128)
        self._weight_init()
    def _weight_init(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonl
            # inearity='relu')
            if isinstance(m, nn.Conv2d):
                #print(m.weight.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        # Zero Gamma
        # count = 0
        # for m in self.modules():
        #     if isinstance(m, HPM_ConvBlock):
        #         for name, x in m.named_modules():
        #             if name == "bn3":
        #                 count += 1
        #                 x.weight.data.zero_()
        # print("count = ", count)
    def forward(self, x):
        outputs = []
        outputs_angle = []
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
            if self.training:
                outputs_angle.append(self.aux_net(x))
            
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
        if self.training:
            return outputs, outputs_angle
        else:
            return outputs



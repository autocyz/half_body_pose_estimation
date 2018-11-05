"""
@author: autocyz
@contact: autocyz@163.com
@file: peleenet.py
@function: PeleePoseNet
@time: 18-10-30
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models.resnet
import torchstat

def conv_bn_relu(inp, oup, kernel_size=3, stride=1, pad=1, use_relu=True):
    f = [nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
        nn.BatchNorm2d(oup)]
    if use_relu:
        f.append(nn.ReLU(inplace=True))
    return nn.Sequential(*f)


class StemBlock(nn.Module):
    def __init__(self, inp=3, num_init_features=32, init_stride=2):
        super(StemBlock, self).__init__()

        self.stem_1 = conv_bn_relu(inp, num_init_features, 3, init_stride, 1)

        self.stem_2a = conv_bn_relu(num_init_features, int(num_init_features / 2), 1, 1, 0)

        self.stem_2b = conv_bn_relu(int(num_init_features / 2), num_init_features, 3, 2, 1)

        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem_3 = conv_bn_relu(num_init_features * 2, num_init_features, 1, 1, 0)

    def forward(self, x):
        stem_1_out = self.stem_1(x)

        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2p_out = self.stem_2p(stem_1_out)
        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growth_rate):
        super(DenseBlock, self).__init__()

        self.cb1_a = conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb1_b = conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)

        self.cb2_a = conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb2_b = conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)
        self.cb2_c = conv_bn_relu(growth_rate, growth_rate, 3, 1, 1)

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = torch.cat((x, cb1_b_out, cb2_c_out), 1)

        return out


class TransitionBlock(nn.Module):
    def __init__(self, inp, oup, with_pooling=True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(conv_bn_relu(inp, oup, 1, 1, 0),
                                    nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.tb = conv_bn_relu(inp, oup, 1, 1, 0)

    def forward(self, x):
        out = self.tb(x)
        return out


class PeleePoseNet(nn.Module):
    def __init__(self):
        super(PeleePoseNet, self).__init__()
        self.stage_first = nn.Sequential()
        self.stage_cpm = nn.Sequential()
        self.stage_paf = nn.Sequential()

        self.stage_first.add_module('stem_1', StemBlock(3, 32, init_stride=2))
        self.stage_first.add_module('stem_2', StemBlock(32, 32, init_stride=1))

        for i in range(7):
            self.stage_cpm.add_module('cpm_%d'%i, DenseBlock(32, 32, 16))
            self.stage_cpm.add_module('cpm_%d_conv'%i, conv_bn_relu(64, 32, 3, 1, 1, True))
            self.stage_paf.add_module('paf_%d' % i, DenseBlock(32, 32, 16))
            self.stage_paf.add_module('paf_%d_conv' % i, conv_bn_relu(64, 32, 3, 1, 1, True))

        self.stage_last_cpm = nn.Conv2d(32, 5, 1, 1, 0)
        self.stage_last_paf = nn.Conv2d(32, 6, 1, 1, 0)

    def forward(self, x):
        feature = self.stage_first(x)
        cpm = self.stage_cpm(feature)
        paf = self.stage_paf(feature)

        cpm = self.stage_last_cpm(cpm)
        paf = self.stage_last_paf(paf)

        return cpm, paf

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




if __name__ == '__main__':
    from torchviz import make_dot
    net = PeleePoseNet()
    # compute model size and FLOPs
    torchstat.stat(net, (3, 368, 368))

    # draw model graph
    # x = torch.randn(1, 3, 368, 368).requires_grad_(True)
    # y = net(x)
    # cc = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    # cc.view('PeleePoseNet')

    # compute model params size
    # summary(net, (3, 368, 368), device='cpu')

















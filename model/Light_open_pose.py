"""
@author: autocyz
@contact: autocyz@163.com
@file: Light_open_pose.py
@function: 
@time: 18-12-17
"""

import torch
import torch.nn as nn


def conv_bn_relu(inp, oup, kernel_size=3, stride=1, pad=1, group=1, use_bn=True, use_relu=True, bias=False, dilation=1):
    f = [nn.Conv2d(inp, oup, kernel_size, stride, pad, groups=group, bias=bias, dilation=dilation)]
    if use_bn:
        f.append(nn.BatchNorm2d(oup))
    if use_relu:
        f.append(nn.ReLU(inplace=True))
    return nn.Sequential(*f)


def depth_wise_conv(inp, oup, kernel_size=3, stride=2, pad=1, dilation=1, use_bias=False, use_relu=True, use_BN=True):
    f = [
        nn.Conv2d(inp, inp, kernel_size, stride, pad, groups=inp, bias=use_bias, dilation=dilation)
    ]
    if use_BN:
        f.append(nn.BatchNorm2d(inp))
    if use_relu:
        f.append(nn.ReLU(inplace=True))

    f.append(nn.Conv2d(inp, oup, 1, 1, 0, groups=1, bias=use_bias))
    if use_BN:
        f.append(nn.BatchNorm2d(oup))
    if use_relu:
        f.append(nn.ReLU(inplace=True))

    return nn.Sequential(*f)


class BackBone(nn.Module):
    def __init__(self, backbone_param):
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, groups=1, bias=False)
        self.depth_wise_convs_list = []
        for param in backbone_param:
            self.depth_wise_convs_list.append(depth_wise_conv(*param))

        self.depth_wise_convs = nn.Sequential(*self.depth_wise_convs_list)
        self.convn = nn.Conv2d(512, 128, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depth_wise_convs(x)
        x = self.convn(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inp, oup,):
        super(ResBlock, self).__init__()
        self.base = nn.Sequential(conv_bn_relu(inp, oup, 1, 1, 0, group=1, use_bn=False, use_relu=True, bias=True))
        self.branch = nn.Sequential(
            conv_bn_relu(oup, oup, 3, 1, 1, use_bn=True, use_relu=True, group=1, bias=True),
            conv_bn_relu(oup, oup, 3, 1, 2, use_bn=True, use_relu=True, group=1, bias=True, dilation=2)
        )

    def forward(self, input):
        input = self.base(input)
        res = self.branch(input)

        return res + input


class Stage(nn.Module):
    def __init__(self, n_cpm, n_paf):
        super(Stage, self).__init__()
        self.layers_1 = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ELU(inplace=True))

        self.conv4_4 = conv_bn_relu(128, 128, 3, 1, 1, use_bn=False, use_relu=True, group=1, bias=True)
        self.cpm_paf = nn.Sequential(
            conv_bn_relu(128, 128, 3, 1, 1, use_bn=False, use_relu=True, group=1, bias=True),
            conv_bn_relu(128, 128, 3, 1, 1, use_bn=False, use_relu=True, group=1, bias=True),
            conv_bn_relu(128, 128, 3, 1, 1, use_bn=False, use_relu=True, group=1, bias=True)
        )
        self.cpm = nn.Sequential(
            conv_bn_relu(128, 128, 1, 1, 0, use_bn=False, use_relu=True, group=1, bias=True),
            nn.Conv2d(128, n_cpm, 1, 1, 0)
        )
        self.paf = nn.Sequential(
            conv_bn_relu(128, 128, 1, 1, 0, use_bn=False, use_relu=True, group=1, bias=True),
            nn.Conv2d(128, n_paf, 1, 1, 0)
        )

    def forward(self, x):
        y = self.layers_1(x)
        y = y + x
        y = self.conv4_4(y)
        z = self.cpm_paf(y)
        z_cpm = self.cpm(z)
        z_paf = self.paf(z)
        return torch.cat((y, z_cpm, z_paf), 1), z_cpm, z_paf


class LightOpenPose(nn.Module):
    def __init__(self):
        super(LightOpenPose, self).__init__()
        self.n_cpm = 5
        self.n_paf = 6
        self.n_resblock = 5
        backbone_param = [
            # inp, out, kernel, stride, pad, dilation
            [  32,  64,      3,      1,   1,        1 ],  #conv2_1
            [  64, 128,      3,      2,   1,        1 ],  #conv2_2
            [ 128, 128,      3,      1,   1,        1 ],  #conv3_1
            [ 128, 256,      3,      2,   1,        1 ],  #conv3_2
            [ 256, 256,      3,      1,   1,        1 ],  #conv4_1
            [ 256, 512,      3,      1,   1,        1 ],  #conv4_2
            [ 512, 512,      3,      1,   2,        2 ],  #conv5_1
            [ 512, 512,      3,      1,   1,        1 ],  #conv5_2
            [ 512, 512,      3,      1,   1,        1 ],  #conv5_3
            [ 512, 512,      3,      1,   1,        1 ],  #conv5_4
            [ 512, 512,      3,      1,   1,        1 ],  #conv5_5
        ]
        self.backbone = BackBone(backbone_param)
        self.stage1 = Stage(self.n_cpm, self.n_paf)
        self.res_blocks = nn.Sequential(*([ResBlock(128+self.n_cpm+self.n_paf, 128)] + [ResBlock(128, 128)]*(self.n_resblock-1)))
        self.out_cpm = nn.Sequential(
            conv_bn_relu(128, 128, 1, 1, 0, use_bn=False, use_relu=True),
            nn.Conv2d(128, self.n_cpm, 1, 1, 0)
        )
        self.out_paf = nn.Sequential(
            conv_bn_relu(128, 128, 1, 1, 0, use_bn=False, use_relu=True),
            nn.Conv2d(128, self.n_paf, 1, 1, 0)
        )

    def forward(self, x):
        x = self.backbone(x)
        z, cpm_1, paf_1 = self.stage1(x)
        z = self.res_blocks(z)
        cpm_2 = self.out_cpm(z)
        paf_2 = self.out_paf(z)
        return cpm_1, paf_1, cpm_2, paf_2



if __name__ == "__main__":


    from torchviz import make_dot
    import torchstat
    from torchsummary import summary


    net = LightOpenPose()


    # compute model size and FLOPs
    # torchstat.stat(net, (3, 368, 368))

    # draw model graph
    x = torch.randn(1, 3, 368, 368).requires_grad_(True)
    y = net(x)
    cc = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
    cc.view('PeleePoseNet')

    # compute model params size
    # summary(net, (3, 368, 368), device='cpu')




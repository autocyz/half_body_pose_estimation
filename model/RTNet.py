import torch.nn as nn
from torchsummary import summary
import numpy as np
import torch
from torchviz import make_dot

def conv_relu(inp, out, kernel, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel, stride=stride, padding=pad),
        nn.ReLU(inplace=True)
    )


def stage_first(ratio):
    return nn.Sequential(       # default input shape(320, 240)
        conv_relu(3, int(64/ratio), 3, 1, 1),
        conv_relu(int(64/ratio), int(64/ratio), 3, 1, 1),
        nn.MaxPool2d(2, 2),          # shape = input_shape/2
        conv_relu(int(64/ratio), int(128/ratio), 3, 1, 1),
        conv_relu(int(128/ratio), int(128/ratio), 3, 1, 1),
        nn.MaxPool2d(2, 2),          # shape = input_shape/4
        conv_relu(int(128/ratio), int(256/ratio), 3, 1, 1),
        conv_relu(int(256/ratio), int(256/ratio), 3, 1, 1),
        conv_relu(int(256/ratio), int(256/ratio), 3, 1, 1),
        conv_relu(int(256/ratio), int(256/ratio), 3, 1, 1),
        nn.MaxPool2d(2, 2),          # shape = input_shape/8
        conv_relu(int(256/ratio), int(512/ratio), 3, 1, 1),
        conv_relu(int(512/ratio), int(512/ratio), 3, 1, 1),
        conv_relu(int(512/ratio), int(256/ratio), 3, 1, 1),
        conv_relu(int(256/ratio), int(128/ratio), 3, 1, 1),
    )


def stage_block_CPM(first_ratio, ratio):
    return nn.Sequential(
        conv_relu(int(128/first_ratio), int(128/ratio), 3, 1, 1),
        conv_relu(int(128/ratio), int(128/ratio), 3, 1, 1),
        conv_relu(int(128/ratio), int(128/ratio), 3, 1, 1),
        conv_relu(int(128/ratio), int(512/ratio), 1, 1, 0),
        nn.Conv2d(int(512/ratio), 5, 1, 1, 0)
    )


def stage_block_PAF(first_ratio, ratio):
    return nn.Sequential(
        conv_relu(int(128/first_ratio), int(128/ratio), 3, 1, 1),
        conv_relu(int(128/ratio), int(128/ratio), 3, 1, 1),
        conv_relu(int(128/ratio), int(128/ratio), 3, 1, 1),
        conv_relu(int(128/ratio), int(512/ratio), 1, 1, 0),
        nn.Conv2d(int(512/ratio), 6, 1, 1, 0)
    )


def stage_last_CPM(first_ratio, ratio):
    return nn.Sequential(
        conv_relu(int(128/first_ratio)+5+6, int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 1, 1, 0),
        # conv_relu(128, 5, 1, 1, 0)
        nn.Conv2d(int(128/ratio), 5, 1, 1, 0)
    )


def stage_last_PAF(first_ratio, ratio):
    return nn.Sequential(
        conv_relu(int(128/first_ratio)+5+6, int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 7, 1, 3),
        conv_relu(int(128/ratio), int(128/ratio), 1, 1, 0),
        # conv_relu(128, 6, 1, 1, 0),
        nn.Conv2d(int(128/ratio), 6, 1, 1, 0)
    )


class RTNet(nn.Module):
    def __init__(self):
        super(RTNet, self).__init__()
        self.stage_first = stage_first(1)
        self.stage_block_cpm = stage_block_CPM(1, 1)
        self.stage_block_paf = stage_block_PAF(1, 1)
        self.stage_last_cpm = stage_last_CPM(1, 1)
        self.stage_last_paf = stage_last_PAF(1, 1)

    def forward(self, input):
        y = self.stage_first(input)
        cpm_1 = self.stage_block_cpm(y)
        paf_1 = self.stage_block_paf(y)
        y = torch.cat((y, cpm_1, paf_1), 1)
        cpm = self.stage_last_cpm(y)
        paf = self.stage_last_paf(y)
        # y = torch.cat((self.stage_last_cpm(y), self.stage_last_paf(y)), 1)
        return cpm_1, paf_1, cpm, paf


class RTNet_Half(nn.Module):
    def __init__(self):
        super(RTNet_Half, self).__init__()
        self.stage_first = stage_first(4)
        self.stage_block_cpm = stage_block_CPM(4, 4)
        self.stage_block_paf = stage_block_PAF(4, 4)
        self.stage_last_cpm = stage_last_CPM(4, 4)
        self.stage_last_paf = stage_last_PAF(4, 4)

    def forward(self, input):
        y = self.stage_first(input)
        cpm_1 = self.stage_block_cpm(y)
        paf_1 = self.stage_block_paf(y)
        y = torch.cat((y, cpm_1, paf_1), 1)
        cpm = self.stage_last_cpm(y)
        paf = self.stage_last_paf(y)
        # y = torch.cat((self.stage_last_cpm(y), self.stage_last_paf(y)), 1)
        return cpm_1, paf_1, cpm, paf


if __name__ == '__main__':
    import torchstat
    # compute model size and FLOPs
    model = RTNet_Half()
    model = RTNet()
    torchstat.stat(model, (3, 368, 368))

    # plot model graph 
    # x = torch.randn(1, 3, 368, 368).requires_grad_(True)
    # y = model(x)
    # cc = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))

    # compute model params size
    # summary(model, (3, 368, 368), device='cpu')


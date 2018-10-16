import torch.nn as nn
from torchsummary import summary
import numpy as np
import torch


def conv_relu(inp, out, kernel, stride=1, pad=1):
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel, stride=stride, padding=pad),
        nn.ReLU(inplace=True)
    )


def stage_first():
    return nn.Sequential(       # default input shape(320, 240)
        conv_relu(3, 64, 3, 1, 1),
        conv_relu(64, 64, 3, 1, 1),
        nn.MaxPool2d(2, 2),          # shape = input_shape/2
        conv_relu(64, 128, 3, 1, 1),
        conv_relu(128, 128, 3, 1, 1),
        nn.MaxPool2d(2, 2),          # shape = input_shape/4
        conv_relu(128, 256, 3, 1, 1),
        conv_relu(256, 256, 3, 1, 1),
        conv_relu(256, 256, 3, 1, 1),
        conv_relu(256, 256, 3, 1, 1),
        nn.MaxPool2d(2, 2),          # shape = input_shape/8
        conv_relu(256, 512, 3, 1, 1),
        conv_relu(512, 512, 3, 1, 1),
        conv_relu(512, 256, 3, 1, 1),
        conv_relu(256, 128, 3, 1, 1),
    )


def stage_block_CPM():
    return nn.Sequential(
        conv_relu(128, 128, 3, 1, 1),
        conv_relu(128, 128, 3, 1, 1),
        conv_relu(128, 128, 3, 1, 1),
        conv_relu(128, 512, 1, 1, 0),
        nn.Conv2d(512, 5, 1, 1, 0)
    )


def stage_block_PAF():
    return nn.Sequential(
        conv_relu(128, 128, 3, 1, 1),
        conv_relu(128, 128, 3, 1, 1),
        conv_relu(128, 128, 3, 1, 1),
        conv_relu(128, 512, 1, 1, 0),
        nn.Conv2d(512, 6, 1, 1, 0)
    )



def stage_last_CPM():
    return nn.Sequential(
        conv_relu(128+5+6, 128, 7, 1, 3),
        conv_relu(128, 128, 7, 1, 3),
        conv_relu(128, 128, 7, 1, 3),
        conv_relu(128, 128, 7, 1, 3),
        conv_relu(128, 128, 7, 1, 3),
        conv_relu(128, 128, 1, 1, 0),
        # conv_relu(128, 5, 1, 1, 0)
        nn.Conv2d(128, 5, 1, 1, 0)
    )


def stage_last_PAF():
    return nn.Sequential(
        conv_relu(128+5+6, 128, 7, 1, 3),
        conv_relu(128, 128, 7, 1, 3),
        conv_relu(128, 128, 7, 1, 3),
        conv_relu(128, 128, 7, 1, 3),
        conv_relu(128, 128, 7, 1, 3),
        conv_relu(128, 128, 1, 1, 0),
        # conv_relu(128, 6, 1, 1, 0),
        nn.Conv2d(128, 6, 1, 1, 0)
    )


class RTNet(nn.Module):
    def __init__(self):
        super(RTNet, self).__init__()
        self.stage_first = stage_first()
        self.stage_block_cpm = stage_block_CPM()
        self.stage_block_paf = stage_block_PAF()
        self.stage_last_cpm = stage_last_CPM()
        self.stage_last_paf = stage_last_PAF()

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
    model = RTNet().cpu()
    summary(model, (3, 368, 368), device='cpu')


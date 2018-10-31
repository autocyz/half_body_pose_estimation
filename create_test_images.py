"""
@author: autocyz
@contact: autocyz@163.com
@file: create_test_images.py
@function: create test result images with joints and limb
@time: 18-10-25
"""


import torch
import cv2
import os
import random
import numpy as np
from model.RTNet import RTNet, RTNet_Half
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from pose_decode import  decode_pose
import time


model_path = 'result/checkpoint/1030_1/epoch_8_0.028339.cpkt'
images_path = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_images_20180103'
result_path = 'result/image/'
use_gpu = False
is_resize = True
param = {'thre1': 0.3, 'thre2': 0.05, 'thre3': 0.5}


# net = RTNet()
net = RTNet_Half()
if use_gpu:
    net = net.cuda()
net.load_state_dict(torch.load(model_path))
net.eval()


with torch.no_grad():
    image_names = os.listdir(images_path)
    image_names = sorted(image_names)
    time
    for i in range(1000):
        name = image_names[i]
        a = os.path.join(images_path, name)
        src_img = cv2.imread(os.path.join(images_path, name))
        if is_resize:
            src_img = cv2.resize(src_img, dsize=(368, 368))
        print('\nimage: {} size：{}'.format(name, src_img.shape))

        T_start = time.clock()
        img = np.transpose(src_img, [2, 0, 1])
        img = np.asarray(img, dtype=np.float32) / 255.
        img = torch.from_numpy(img)
        img = torch.Tensor.unsqueeze(img, 0)
        if use_gpu:
            img = img.cuda()
        print('net inference.....')
        T_data_process = time.clock()
        _, _, cpm, paf = net(img)
        T_net_inference = time.clock()
        print('.....net inference')
        heatmaps = cpm.cpu().data.numpy().transpose(0, 2, 3, 1)
        pafs = paf.cpu().data.numpy().transpose(0, 2, 3, 1)
        canvas, joint_list, person_to_joint_assoc = decode_pose(src_img, param, heatmaps[0], pafs[0])
        T_result_process = time.clock()

        print('data_process: {}\nnet_inference: {}\nresult_process: {}'
              .format(T_data_process - T_start,
                      T_net_inference - T_data_process,
                      T_result_process - T_net_inference))

        cv2.imwrite(os.path.join(result_path, name), canvas )


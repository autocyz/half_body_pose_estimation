import torch
import cv2
import numpy as np
from model import RTNet
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from pose_decode import  decode_pose
import time

use_gpu = False
model_path = 'result/checkpoint/1019/epoch_1.cpkt'
# image_path = 'data/4.jpg'
image_path = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_test_a_20180103/' \
             'keypoint_test_a_images_20180103/0ca92c6ec20192d40b5e7b7199af147ec2c36033.jpg'
net = RTNet()
if use_gpu:
    net = net.cuda()

writer = SummaryWriter(log_dir='data/')
src_img = cv2.imread(image_path)
param = {'thre1': 0.1, 'thre2': 0.00, 'thre3': 0.5}
# src_img = cv2.resize(src_img, dsize=(368, 368))
print('image_size：', src_img.shape)
with torch.no_grad():
    net.load_state_dict(torch.load(model_path))
    net.eval()

    T_start = time.clock()
    img = np.transpose(src_img, [2, 0, 1])
    img = np.asarray(img, dtype=np.float32) / 255.
    img = torch.from_numpy(img)
    img = torch.Tensor.unsqueeze(img, 0)
    if use_gpu:
        img = img.cuda()
    T_data_process = time.clock()
    _, _, cpm, paf = net(img)
    T_net_inference = time.clock()

    heatmaps = cpm.cpu().data.numpy().transpose(0, 2, 3, 1)
    pafs = paf.cpu().data.numpy().transpose(0, 2, 3, 1)
    heatmap_total = np.sum(heatmaps[:, :, :, :-1], axis=3)
    heatmap_total = np.transpose(heatmap_total, [1, 2, 0])

    cc = np.expand_dims(np.transpose(heatmaps[0, :, :, :], [2, 0, 1]), 1)
    writer.add_image('heatmap_target', cc)
    cc_max = np.max(np.max(cc, axis=-1), axis=-1).reshape(cc.shape[0], 1, 1, 1)
    cc_min = np.min(np.min(cc, axis=-1), axis=-1).reshape(cc.shape[0], 1, 1, 1)
    cc_diff = cc_max - cc_min
    cc = (cc - cc_min) / cc_diff
    writer.add_image('heatmap_target_norm', cc)

    canvas, joint_list, person_to_joint_assoc = decode_pose(src_img, param, heatmaps[0], pafs[0])
    T_result_process = time.clock()

    print('data_process: {}\nnet_inference: {}\nresult_process: {}'
          .format(T_data_process - T_start,
                  T_net_inference - T_data_process,
                  T_result_process - T_net_inference))
    plt.figure(1)
    plt.imshow(canvas)
    plt.show()


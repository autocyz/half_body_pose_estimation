"""
author: autocyz
date: 2018.10.12
function: train half body pose
"""

import torch
import os
import numpy as np
from params import params_transform
from dataset import AIChallenge
from model import RTNet

from torch.utils.data.dataloader import DataLoader
from torch.optim.sgd import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from utils import save_params, get_lr

train_image_path = '/mnt/data/dataset/PoseData/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'
train_anno_file = '/mnt/data/dataset/PoseData/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
val_image_path = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
val_anno_file = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
mode_save_path = './result/checkpoint/'


trainset = AIChallenge(train_image_path, train_anno_file, params_transform)
valset = AIChallenge(val_image_path, val_anno_file, params_transform)

train_loader = DataLoader(trainset,
                          batch_size=params_transform['batch_size'],
                          shuffle=True,
                          num_workers=params_transform['num_workers']
                          )
val_loader = DataLoader(valset,
                        batch_size=params_transform['batch_size'],
                        shuffle=True,
                        num_workers=params_transform['num_workers']
                        )

net = RTNet().cuda()

# optimizer = SGD(net.parameters(),
#                 lr=params_transform['learning_rate'],
#                 momentum=params_transform['momentum'],
#                 weight_decay=params_transform['weight_decay'],
#                 nesterov=params_transform['nesterov']
#                 )
optimizer = torch.optim.Adam(net.parameters(), lr=params_transform['learning_rate'])
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=1e-4)
criterion = torch.nn.MSELoss(size_average=True, reduce=True).cuda()

date = '1012'
writer = SummaryWriter(log_dir='./result/logdir/' + date)
model_path = os.path.join('./result/checkpoint/', date)
if not os.path.exists(model_path):
    os.mkdir(model_path)

# writer.add_graph(net, torch.ones(16, 3, 368, 368))

best_loss = np.inf
for epoch in range(params_transform['epoch_num']):
    for i, (img, heatmaps, pafs) in enumerate(train_loader):
        target = torch.cat((heatmaps, pafs), 1)
        target = target.float()
        img = img.cuda()
        target = target.cuda()
        predict = net(img)
        loss = criterion(target, predict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer some train information
        total_iter = epoch * len(val_loader) + i
        writer.add_scalar('train_loss', loss.data.cpu(), total_iter)
        if total_iter % 10 == 0:
            writer.add_image('heatmap_target', torch.unsqueeze(heatmaps[0, :, :, :], 1))
            writer.add_image('heatmap_predit', torch.unsqueeze(predict[0, 0:5, :, :], 1))
            writer.add_image('paf_target', torch.unsqueeze(pafs[0, :, :, :], 1))
            writer.add_image('paf_predit', torch.unsqueeze(predict[0, 5:, :, :], 1))

        print('Epoch [{}/{}], Step [{}/{}], Lr [{}], Loss: {:.4f}'.
              format(epoch+1, params_transform['epoch_num'], total_iter,
                     len(train_loader)*params_transform['epoch_num'], get_lr(optimizer), loss.item()))
    # lr_scheduler.step(loss)
    if loss < best_loss:
        best_loss = loss
        torch.save(net.state_dict(), model_path + 'eppch_%d.cpkt'%epoch)

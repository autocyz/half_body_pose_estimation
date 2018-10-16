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
from torch.optim.lr_scheduler import StepLR
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

torch.backends.cudnn.benchmark = True
net = RTNet().cuda()
if params_transform['pretrain_model']:
    print("loading pre_trained model :", params_transform['pretrain_model'])
    net.load_state_dict(torch.load(params_transform['pretrain_model']))
    print("loading over")

# optimizer = SGD(net.parameters(),
#                 lr=params_transform['learning_rate'],
#                 momentum=params_transform['momentum'],
#                 weight_decay=params_transform['weight_decay']
#                 # nesterov=params_transform['nesterov']
#                 )
optimizer = torch.optim.Adam(net.parameters(), lr=params_transform['learning_rate'])
# lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=1e-4)
lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
criterion = torch.nn.MSELoss().cuda()

date = '1016'
writer = SummaryWriter(log_dir='./result/logdir/' + date)
model_path = os.path.join('./result/checkpoint/', date)
if not os.path.exists(model_path):
    os.mkdir(model_path)

save_params(model_path, 'parameter', params_transform)
# writer.add_graph(net, torch.ones(16, 3, 368, 368))


best_loss = np.inf
for epoch in range(params_transform['epoch_num']):
    # train
    lr_scheduler.step()
    net.train()
    for i, (img, heatmaps, pafs) in enumerate(train_loader):
        heatmaps = heatmaps.float().cuda()
        pafs = pafs.float().cuda()
        img = img.cuda()

        cpm_1, paf_1, cpm_p, paf_p = net(img)

        loss_cpm1 = criterion(heatmaps, cpm_1)
        loss_paf1 = criterion(pafs, paf_1)
        loss_cpm = criterion(heatmaps, cpm_p)
        loss_paf = criterion(pafs, paf_p)
        total_loss = loss_cpm + loss_paf + loss_cpm1 + loss_paf1
        # total_loss = loss_cpm + loss_paf

        optimizer.zero_grad()
        total_loss.backward()
        # loss_paf.backward()
        optimizer.step()

        # writer some train information
        total_iter = epoch * len(train_loader) + i
        writer.add_scalar('train_loss_cpm', loss_cpm.item(), total_iter)
        writer.add_scalar('train_loss_paf', loss_paf.item(), total_iter)
        writer.add_scalars('all_loss', {'cpm': loss_cpm.item(), 'paf': loss_paf.item(), 'total': total_loss.item()}, total_iter)

        if total_iter % params_transform['display'] == 0:
            writer.add_image('cpm_1', torch.unsqueeze(cpm_1[0], 1))
            writer.add_image('heatmap_target', torch.unsqueeze(heatmaps[0, :, :, :], 1))
            writer.add_image('heatmap_predit', torch.unsqueeze(cpm_p[0], 1))
            writer.add_image('paf_target', torch.unsqueeze(pafs[0, :, :, :], 1))
            writer.add_image('paf_predit', torch.unsqueeze(paf_p[0], 1))

        print('Epoch [{}/{}]\tStep [{}/{}]\tLr [{}]'
              '\tloss_cpm_1 {:.3f}\tloss_paf_1 {:.3f}\tloss_cpm {:.3f}'
              '\tloss_paf {:.3f} \tTotal_loss {:.3f}'.
              format(epoch+1, params_transform['epoch_num'], total_iter,
                     len(train_loader)*params_transform['epoch_num'], get_lr(optimizer),
                     loss_cpm1.item(), loss_paf1.item(), loss_cpm.item(), loss_paf.item(), total_loss.item()))

        if i == 2:
            break
    # validation
    net.eval()
    val_loss = 0.
    with torch.no_grad():
        for i, (img, heatmaps, pafs) in enumerate(val_loader):
            heatmaps = heatmaps.float().cuda()
            pafs = pafs.float().cuda()
            img = img.cuda()
            cpm_1, paf_1, cpm_p, paf_p = net(img)

            loss_cpm1 = criterion(heatmaps, cpm_1)
            loss_paf1 = criterion(pafs, paf_1)
            loss_cpm = criterion(heatmaps, cpm_p)
            loss_paf = criterion(pafs, paf_p)
            total_loss = loss_cpm + loss_paf + loss_cpm1 + loss_paf1
            print(loss_cpm1, loss_paf1, loss_cpm, loss_paf, total_loss)
            val_loss += total_loss

            if i % params_transform['display'] == 0:
                writer.add_image('cpm_1', torch.unsqueeze(cpm_1[0], 1))
                writer.add_image('heatmap_target', torch.unsqueeze(heatmaps[0, :, :, :], 1))
                writer.add_image('heatmap_predit', torch.unsqueeze(cpm_p[0], 1))
                writer.add_image('paf_target', torch.unsqueeze(pafs[0, :, :, :], 1))
                writer.add_image('paf_predit', torch.unsqueeze(paf_p[0], 1))

    val_loss = val_loss / len(val_loader)
    print('epoch [{}] val_loss [{:.4f}]'.format(epoch, val_loss))
    writer.add_scalar('val_loss', val_loss, epoch)
    # lr_scheduler.step(loss)
    if val_loss.item() < best_loss:
        best_loss = val_loss.item()
        torch.save(net.state_dict(), os.path.join(model_path, 'epoch_%d.cpkt'%epoch))

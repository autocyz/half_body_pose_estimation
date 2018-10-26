"""
@author: autocyz
@contact: autocyz@163.com
@file: train.py
@function: train model
@time: 18-10-15
"""

import torch
import os
import time
import numpy as np
from params import params_transform
from dataset import AIChallenge
from model import RTNet

from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from utils import save_params, get_lr


def train(train_loader, net, criterion, optimizer, epoch, loader_info=''):
    for i, (img, heatmaps, pafs) in enumerate(train_loader):
        time0 = time.clock()
        heatmaps = heatmaps.float().cuda()
        pafs = pafs.float().cuda()
        img = img.cuda()

        time1 = time.clock()
        cpm_1, paf_1, cpm_p, paf_p = net(img)
        time2 = time.clock()

        loss_cpm1 = criterion(heatmaps, cpm_1)
        loss_paf1 = criterion(pafs, paf_1)
        loss_cpm = criterion(heatmaps, cpm_p)
        loss_paf = criterion(pafs, paf_p)
        total_loss = loss_cpm + loss_paf + loss_cpm1 + loss_paf1
        # total_loss = loss_cpm + loss_paf

        time3 = time.clock()
        optimizer.zero_grad()
        total_loss.backward()
        # loss_paf.backward()
        optimizer.step()
        time4 = time.clock()

        # writer some train information
        total_iter = epoch * len(train_loader) + i
        writer.add_scalar('train_loss_cpm', loss_cpm.item(), total_iter)
        writer.add_scalar('train_loss_paf', loss_paf.item(), total_iter)
        writer.add_scalars('all_loss', {'cpm': loss_cpm.item(), 'paf': loss_paf.item(), 'total': total_loss.item()}, total_iter)

        print('Epoch [{}/{}]\tStep [{}/{} {}]\tLr [{}]'
              '\tloss_cpm_1 {:.3f}\tloss_paf_1 {:.3f}\tloss_cpm {:.3f}'
              '\tloss_paf {:.3f} \tTotal_loss {:.3f}\n'.
              format(epoch, params_transform['epoch_num'], i,
                     len(train_loader), loader_info, get_lr(optimizer),
                     loss_cpm1.item(), loss_paf1.item(), loss_cpm.item(), loss_paf.item(), total_loss.item()))

        if total_iter % params_transform['display'] == 0:
            writer.add_image('heatmap_target', torch.unsqueeze(heatmaps[0, :, :, :], 1))
            writer.add_image('paf_target', torch.unsqueeze(pafs[0, :, :, :], 1))

            heatmaps_norm = torch.Tensor.unsqueeze(heatmaps[0, :, :, :], 1)
            heatmaps_norm_max = torch.Tensor.max(torch.Tensor.max(heatmaps_norm, dim=-1)[0], dim=-1)[0].\
                reshape(heatmaps_norm.shape[0], 1, 1, 1)
            heatmaps_norm_min = torch.Tensor.min(torch.Tensor.min(heatmaps_norm, dim=-1)[0], dim=-1)[0].\
                reshape(heatmaps_norm.shape[0], 1, 1, 1)
            heatmaps_norm_diff = heatmaps_norm_max - heatmaps_norm_min
            heatmaps_norm = (heatmaps_norm - heatmaps_norm_min) / heatmaps_norm_diff
            writer.add_image('heatmap_predit', heatmaps_norm)

            pafs_norm = torch.Tensor.unsqueeze(pafs[0, :, :, :], 1)
            pafs_norm_max = torch.Tensor.max(torch.Tensor.max(pafs_norm, dim=-1)[0], dim=-1)[0].\
                reshape(pafs_norm.shape[0], 1, 1, 1)
            pafs_norm_min = torch.Tensor.min(torch.Tensor.min(pafs_norm, dim=-1)[0], dim=-1)[0].\
                reshape(pafs_norm.shape[0], 1, 1, 1)
            pafs_norm_diff = pafs_norm_max - pafs_norm_min
            pafs_norm = (pafs_norm - pafs_norm_min) / pafs_norm_diff
            writer.add_image('paf_predit', pafs_norm, 1)


if __name__ == "__main__":

    train_image_path = '/mnt/data/dataset/PoseData/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902'
    train_anno_file = '/mnt/data/dataset/PoseData/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json'
    val_image_path = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
    val_anno_file = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
    test_image_path_a = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_images_20180103'
    test_anno_file_a = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_annotations_20180103.json'
    test_image_path_b = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_test_b_20180103/keypoint_test_b_images_20180103'
    test_anno_file_b = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_test_b_20180103/keypoint_test_b_annotations_20180103.json'
    mode_save_path = './result/checkpoint/'

    print('loading trainset')
    trainset = AIChallenge(train_image_path, train_anno_file, params_transform)
    print('loading valset')
    valset = AIChallenge(val_image_path, val_anno_file, params_transform)
    print('loading testset a')
    testset_a = AIChallenge(test_image_path_a, test_anno_file_a, params_transform)
    print('loading testset b')
    testset_b = AIChallenge(test_image_path_b, test_anno_file_b, params_transform)

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
    test_a_loader = DataLoader(testset_a,
                            batch_size=params_transform['batch_size'],
                            shuffle=True,
                            num_workers=params_transform['num_workers']
                            )
    test_b_loader = DataLoader(testset_b,
                            batch_size=params_transform['batch_size'],
                            shuffle=True,
                            num_workers=params_transform['num_workers']
                            )

    torch.backends.cudnn.benchmark = True
    net = RTNet()
    if params_transform['pretrain_model']:
        print("loading pre_trained model :", params_transform['pretrain_model'])
        params_transform['has_checkpoint'] = True
        net.load_state_dict(torch.load(params_transform['pretrain_model']))
        print("loading over")
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=params_transform['learning_rate'], weight_decay=params_transform['weight_decay'])
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=1e-4)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = torch.nn.MSELoss(reduction='elementwise_mean').cuda()

    date = '1026'
    writer = SummaryWriter(log_dir='./result/logdir/' + date)
    model_path = os.path.join('./result/checkpoint/', date)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    params_transform['train_log'] = '在1025的基础上降低十倍学习率'
    save_params(model_path, 'parameter', params_transform)
    # writer.add_graph(net, torch.ones(16, 3, 368, 368))
    # if params_transform['has_checkpoint']:
    #     net.eval()
    #     val_loss = 0.
    #     with torch.no_grad():
    #         for i, (img, heatmaps, pafs) in enumerate(val_loader):
    #             heatmaps = heatmaps.float().cuda()
    #             pafs = pafs.float().cuda()
    #             img = img.cuda()
    #             cpm_1, paf_1, cpm_p, paf_p = net(img)
    #
    #             loss_cpm1 = criterion(heatmaps, cpm_1)
    #             loss_paf1 = criterion(pafs, paf_1)
    #             loss_cpm = criterion(heatmaps, cpm_p)
    #             loss_paf = criterion(pafs, paf_p)
    #             total_loss = loss_cpm + loss_paf + loss_cpm1 + loss_paf1
    #             val_loss += total_loss
    #             if i % 50 == 0:
    #                 print('Step [{}/{}]  current loss [{}]'.format(i, len(val_loader), total_loss))
    #     val_loss = val_loss / len(val_loader)
    #     best_loss = val_loss

    best_loss = np.inf
    for epoch in range(params_transform['epoch_num']):
        # train
        # lr_scheduler.step()
        net.train()
        time4 = time.clock()
        train(test_a_loader, net, criterion, optimizer, epoch, 'test_a')
        train(test_b_loader, net, criterion, optimizer, epoch, 'test_b')
        train(train_loader, net, criterion, optimizer, epoch, 'train')

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
                val_loss += total_loss
                if i % 50 == 0:
                    print('Eval : current loss: ', total_loss)
        val_loss = val_loss / len(val_loader)
        print('epoch [{}] val_loss [{:.4f}]'.format(epoch, val_loss))
        writer.add_scalar('val_loss', val_loss, epoch)
        if val_loss.item() < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), os.path.join(model_path, 'epoch_{}_{:.6f}.cpkt'.format(epoch, val_loss)))

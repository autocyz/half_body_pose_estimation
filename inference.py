import torch
import cv2
import numpy as np
from model.RTNet import RTNet, RTNet_Half
from model.peleenet import PeleePoseNet
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from pose_decode import  decode_pose
import time


def inference(src_img, net, param, use_gpu=False):
    with torch.no_grad():
        T_start = time.time()
        img = np.transpose(src_img, [2, 0, 1])
        img = np.asarray(img, dtype=np.float32) / 255.
        img = torch.from_numpy(img)
        img = torch.Tensor.unsqueeze(img, 0)
        if use_gpu:
            img = img.cuda()
        T_data_process = time.time()
        cpm, paf = net(img)
        T_net_inference = time.time()
        heatmaps = cpm.cpu().data.numpy().transpose(0, 2, 3, 1)
        pafs = paf.cpu().data.numpy().transpose(0, 2, 3, 1)
        canvas, joint_list, person_to_joint_assoc = decode_pose(src_img, param, heatmaps[0], pafs[0])
        T_result_process = time.time()

        print('data_process: {}\nnet_inference: {}\nresult_process: {}'
              .format(T_data_process - T_start,
                      T_net_inference - T_data_process,
                      T_result_process - T_net_inference))

        return canvas


def time_test(net, use_gpu):
    with torch.no_grad() :
        input = np.random.rand(1, 3, 368, 368).astype(np.float32)
        # input = np.random.rand(1, 3, 640, 480).astype(np.float32)
        input = torch.from_numpy(input)
        if use_gpu:
            input = input.cuda()
        start = time.time()
        for i in range(100):
            t1 = time.time()
            cc = net(input)
            t2 = time.time()
            print('every_time: %04d: '%i, t2 - t1)
        end = time.time()
        print('total time: ', end - start)



def showheatmap():
    # use_gpu = False
    use_gpu = True
    model_path = 'result/checkpoint/1030_1/epoch_8_0.028339.cpkt'
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

        T_start = time.time()
        img = np.transpose(src_img, [2, 0, 1])
        img = np.asarray(img, dtype=np.float32) / 255.
        img = torch.from_numpy(img)
        img = torch.Tensor.unsqueeze(img, 0)
        if use_gpu:
            img = img.cuda()
        T_data_process = time.time()
        _, _, cpm, paf = net(img)
        T_net_inference = time.time()

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
        T_result_process = time.time()

        print('data_process: {}\nnet_inference: {}\nresult_process: {}'
              .format(T_data_process - T_start,
                      T_net_inference - T_data_process,
                      T_result_process - T_net_inference))
        plt.figure(1)
        plt.imshow(canvas)
        plt.show()


if __name__ == '__main__':
    import sys
    from model.Light_open_pose import LightOpenPose
    
    # use_gpu = False
    use_gpu = True

    model_path = 'result/checkpoint/1217/epoch_25_0.007546.cpkt'
    # model_path = 'result/checkpoint/1030_1/epoch_8_0.028339.cpkt'
    # model_path = 'result/checkpoint/1016/epoch_5.cpkt'
    # model_path = 'result/checkpoint/1217/epoch_0_0.012472.cpkt'
    
    video_name = '/mnt/data/project/dataset/111.ts'
    # video_name = '/mnt/data/project/dataset/fall_detection/video_fall2.mp4'
    # video_name = '/mnt/data/project/dataset/many_people.avi'
    # video_name = '/mnt/data/project/dataset/peopleCounting/1.mp4'

    if len(sys.argv) > 2:
        model_path = sys.argv[1]
        video_name = sys.argv[2]

    # is_resize = True
    is_resize = False
    
    net = PeleePoseNet()
    # net = RTNet_Half(4)
    # net = RTNet()
    # net = LightOpenPose()
    
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        net = net.cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    print("load model over")

    #=============================
    # test model inference time

    # time_test(net, use_gpu)
    # exit(0)

    #=============================

    # test video human keypoints and limbs, display result
    param = {'thre1': 0.5, 'thre2': 0.05, 'thre3': 0.5}
    capture = cv2.VideoCapture(video_name)
    i = 0
    while(True):
        # i += 1
        # if i%2 == 0:
            # continue
        retval, img = capture.read()
        if not retval:
            break
        if is_resize:
            img = cv2.resize(img, (int(640), int(480)))
            # img = cv2.resize(img, (int(368), int(368)))
            # img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
            print("image size:{}*{} ".format(img.shape[0], img.shape[1]))
        canvas = inference(img, net, param, use_gpu)
        cv2.imshow('result', canvas)

        # print('keynote: ', keynote)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break 

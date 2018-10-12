import torch
import numpy as np
import cv2
import json
from torch.utils.data import DataLoader, Dataset
from skimage.filters import gaussian
import os

''' data set 
train: 
    210000 images
    378374 persons (every image have one person at least)    
val: 
    30000  images
    70402  persons (every image have one person at least)
'''

''' data type 
It has 14 keypoints:
1: r_shoulder  2: r_elbow  3: r_wrist
4: l_shoulder  5: l_elbow  6: l_wrist
7: r_hip       8: r_knee   9: r_ankle
10:l_hip      11: l_knee  12: l_ankle
13:head_top   14: neck

flag: 1(visible) 2(invisible) 3(out of the image or unpredictable)
tip: 1 and 2 means the keypoint has been annotated

{
 'url': 'http://www.sinaimg.cn/dy/slidenews/4_img/2013_24/704_997547_218968.jpg', 
 'image_id': 'd8eeddddcc042544a2570d4c452778b912726720', 
 'keypoint_annotations': 
    {
        'human3': [0, 0, 3, 0, 0, 3, 0, 0, 3, 67, 279, 1, 87, 365, 1, 65, 345, 1, 0, 0, 3, 0, 0, 3, 0, 0, 3, 40, 454, 1, 44, 554, 1, 0, 0, 3, 20, 179, 1, 17, 268, 1], 
        'human2': [444, 259, 1, 474, 375, 2, 451, 459, 1, 577, 231, 1, 632, 396, 1, 589, 510, 1, 490, 538, 1, 0, 0, 3, 0, 0, 3, 581, 535, 2, 0, 0, 3, 0, 0, 3, 455, 78, 1, 486, 205, 1], 
        'human1': [308, 306, 1, 290, 423, 1, 298, 528, 1, 433, 297, 1, 440, 404, 1, 447, 501, 2, 342, 530, 1, 0, 0, 3, 0, 0, 3, 417, 520, 1, 0, 0, 3, 0, 0, 3, 376, 179, 1, 378, 281, 1]
    }, 
 'human_annotations': 
    {
        'human3': [0, 169, 114, 633], 
        'human2': [407, 59, 665, 632], 
        'human1': [265, 154, 461, 632]
    }
}
'''


class AIChallenge(Dataset):
    def __init__(self, image_path, anno_file, params_transform=None):
        self.image_path = image_path
        self.anno_file = anno_file
        self.image_names = []
        self.keypoints = []
        self.human_rects = []
        self.params_transform = params_transform
        self.feature_map_size_x = int(self.params_transform['crop_size_x'] / self.params_transform['feature_map_ratio'])
        self.feature_map_size_y = int(self.params_transform['crop_size_y'] / self.params_transform['feature_map_ratio'])
        self.paf_width_thre = self.params_transform['paf_width_thre']
        self.feature_map_ratio = self.params_transform['feature_map_ratio']
        with open(anno_file) as f:
            annos = json.load(f)
            print(len(annos))
            for anno in annos:
                self.image_names.append(anno['image_id'])
                self.keypoints.append({key: np.asarray(val).reshape(-1, 3)[[0, 3, 12, 13], :] for key, val in anno['keypoint_annotations'].items()})
                self.human_rects.append({key: np.asarray(val).reshape(-1, 4) for key, val in anno['human_annotations'].items()})

        self.numImages = len(self.image_names)

    def create_gauss_map(self, center, size_x, size_y, sigma):
        """
        create gaussian map
        :param center: gaussian map center, (x, y) means the coord
        :param size_x: map size x
        :param size_y: map size y
        :return: confidence map
        """
        x_range = [i for i in range(int(size_x))]
        y_range = [i for i in range(int(size_y))]
        xx, yy = np.meshgrid(x_range, y_range)
        d2 = (xx - center[0])**2 + (yy - center[1])**2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= 4.6052
        confid_map = np.exp(-exponent)
        confid_map = np.multiply(mask, confid_map)
        return confid_map

    def create_paf_map(self, centerA, centerB, size_x, size_y, thresh):
        """
        creat paf vector
        :param centerA:  start point coord
        :param centerB: end point coord
        :param size_x:  map size x
        :param size_y:  map size y
        :param thresh:  width of paf vector
        :return: paf map:
                 mask: mask indicate where should have data
        """
        centerA = centerA.astype(float)
        centerB = centerB.astype(float)
        paf_map = np.zeros((2, size_y, size_x))
        norm = np.linalg.norm(centerB - centerA)
        if norm == 0.0:
            return paf_map
        limb_vec_unit = (centerB - centerA) / norm

        # To make sure not beyond the border of this two points
        min_x = max(int(round(min(centerA[0], centerB[0]) - thresh)), 0)
        max_x = min(int(round(max(centerA[0], centerB[0]) + thresh)), size_x)
        min_y = max(int(round(min(centerA[1], centerB[1]) - thresh)), 0)
        max_y = min(int(round(max(centerA[1], centerB[1]) + thresh)), size_y)

        range_x = list(range(int(min_x), int(max_x), 1))
        range_y = list(range(int(min_y), int(max_y), 1))
        xx, yy = np.meshgrid(range_x, range_y)
        xx = xx.astype(np.int32)
        yy = yy.astype(np.int32)
        ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
        ba_y = yy - centerA[1]
        limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
        mask = limb_width < thresh  # mask is 2D

        paf_map[:, yy, xx] = np.repeat(mask[np.newaxis, :, :], 2, axis=0)
        paf_map[:, yy, xx] *= limb_vec_unit[:, np.newaxis, np.newaxis]
        mask = np.logical_or.reduce((np.abs(paf_map[0, :, :]) > 0, np.abs(paf_map[0, :, :]) > 0))
        return paf_map, mask

    def create_heatmap(self, keypoints):
        # 4 keypoint and one background
        heatmaps = np.zeros((self.params_transform['num_keypoint']+1, self.feature_map_size_y, self.feature_map_size_x))
        for i in range(self.params_transform['num_keypoint']):
            # several person heatmap plus together
            for key, val in keypoints.items():
                # val = 1, 2 mean the point is labeled
                if val[i, 2] < 3:
                    center = val[i, :2] / self.feature_map_ratio
                    gaass_map = self.create_gauss_map(center, self.feature_map_size_x, self.feature_map_size_y, self.params_transform['sigma'])
                    heatmaps[i, :, :] += gaass_map
                    heatmaps[i, :, :][heatmaps[i, :, :] > 1.0] = 1.0
        # add negative heatmap
        heatmaps[-1, :, :] = np.maximum(1 - np.max(heatmaps[:-1, :, :], axis=0), 0.)
        return heatmaps

    def create_paf(self, keypoints):
        # pafs link
        pt1 = [3, 3, 3]
        pt2 = [0, 1, 2]
        pafs = np.zeros((6, self.feature_map_size_y, self.feature_map_size_x))
        for i in range(len(pt1)):
            count = np.zeros((self.feature_map_size_y, self.feature_map_size_x), dtype=np.uint32)
            for key, val in keypoints.items():
                if val[pt1[i], 2] < 3 and val[pt2[i], 2] <3:
                    centerA = val[pt1[i], :2] / self.feature_map_ratio
                    centerB = val[pt2[i], :2] / self.feature_map_ratio
                    paf_map, mask = self.create_paf_map(centerA, centerB, self.feature_map_size_x, self.feature_map_size_y, self.paf_width_thre)
                    pafs[2*i:2*i+2, :, :] = np.multiply(pafs[2*i:2*i+2, :, :], count[np.newaxis, :, :])
                    pafs[2 * i:2 * i + 2, :, :] += paf_map
                    count[mask == True] += 1

                    mask = count == 0
                    count[mask == True] = 1
                    pafs[2*i:2*i+1, :, :] = np.divide(pafs[2*i:2*i+1, :, :], count[np.newaxis, :, :])
                    count[mask == True] = 0
        return pafs

    def get_scale_point(self, img_shape, keypoints):
        height = img_shape[0]
        width = img_shape[1]
        scale_h = self.params_transform['crop_size_y'] / height
        scale_w = self.params_transform['crop_size_x'] / width
        return {key:val*[scale_w, scale_h, 1] for key, val in keypoints.items()}

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img = cv2.imread(os.path.join(self.image_path, image_name + '.jpg'))
        keypoints = self.keypoints[index]
        keypoints = self.get_scale_point(img.shape, keypoints)
        img = cv2.resize(img, (self.params_transform['crop_size_x'], self.params_transform['crop_size_x']))
        # for key, val in keypoints.items():
        #     for point in val:
        #         cv2.circle(img, (int(round(point[0])), int(round(point[1]))), 4, (255, 0, 0))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, [2, 0, 1])
        img = np.asarray(img, dtype=np.float32) / 255

        heatmaps = self.create_heatmap(keypoints)
        pafs = self.create_paf(keypoints)

        return img, heatmaps, pafs

    def __len__(self):
        return self.numImages


if __name__ == "__main__":
    from matplotlib import pyplot
    import random
    from params import params_transform
    from tensorboardX import SummaryWriter

    image_path = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
    anno_file = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
    dataset = AIChallenge(image_path, anno_file, params_transform)

    print('image nums: ', dataset.numImages)
    index = random.randint(0, dataset.numImages-1)
    print(index)
    # index = 1327
    (img, heatmaps, pafs) = dataset.__getitem__(index)
    print(img.shape)
    writer = SummaryWriter(log_dir='./result')
    writer.add_image('image', (img))
    print(np.sum(heatmaps[0,:,:]))
    writer.add_text('image_name', 'image_index:%d'%index)

    for i in range(heatmaps.shape[0]):
        writer.add_image('heatmap%d'%(i+1), np.abs(heatmaps[i,:,:]))

    for i in range(pafs.shape[0]):
        writer.add_image('paf%d'%(i+1), np.abs(pafs[i, :, :]))

    writer.close()




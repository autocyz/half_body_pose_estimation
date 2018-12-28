#!/usr/bin/env python
# coding=utf-8
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation utility for human skeleton system keypoint task.
This python script is used for calculating the final score (mAP) of the test result,
based on your submited file and the reference file containing ground truth.
usage
python keypoint_eval.py --submit SUBMIT_FILEPATH --ref REF_FILEPATH
A test case is provided, submited file is submit.json, reference file is ref.json, test it by:
python keypoint_eval.py --submit ./keypoint_sample_predictions.json \
                        --ref ./keypoint_sample_annotations.json
The final score of the submited result, error message and warning message will be printed.
"""

import json
import time
import argparse
import pprint
import numpy as np


def load_annotations(anno_file, return_dict):
    """Convert annotation JSON file."""

    delta_14_point = 2 * np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709,
                                    0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803,
                                    0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])

    # 0-r-shoulder 3-l-shoulder 12-head 13-neck
    point_4_id = [0, 3, 12, 13]

    annotations = dict()
    annotations['image_ids'] = set([])
    annotations['annos'] = dict()
    annotations['delta'] = delta_14_point[point_4_id]

    try:
        annos = json.load(open(anno_file, 'r'))
    except Exception:
        return_dict['error'] = 'Annotation file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for anno in annos:
        annotations['image_ids'].add(anno['image_id'])
        annotations['annos'][anno['image_id']] = dict()
        annotations['annos'][anno['image_id']]['human_annos'] = anno['human_annotations']
        annotations['annos'][anno['image_id']]['keypoint_annos'] = anno['keypoint_annotations']

    return annotations


def load_predictions(prediction_file, return_dict):
    """Convert prediction JSON file."""
    point_4_id = [0, 3, 12, 13]
    predictions = dict()
    predictions['image_ids'] = []
    predictions['annos'] = dict()
    id_set = set([])

    try:
        preds = json.load(open(prediction_file, 'r'))
    except Exception:
        return_dict['error'] = 'Prediction file does not exist or is an invalid JSON file.'
        exit(return_dict['error'])

    for pred in preds:
        if 'image_id' not in pred.keys():
            return_dict['warning'].append('There is an invalid annotation info, \
                likely missing key \'image_id\'.')
            continue
        if 'keypoint_annotations' not in pred.keys():
            return_dict['warning'].append(pred['image_id'] + \
                                          ' does not have key \'keypoint_annotations\'.')
            continue
        image_id = pred['image_id'].split('.')[0]
        if image_id in id_set:
            return_dict['warning'].append(pred['image_id'] + \
                                          ' is duplicated in prediction JSON file.')
        else:
            id_set.add(image_id)
        predictions['image_ids'].append(image_id)
        predictions['annos'][pred['image_id']] = dict()
        aa = {}
        for key, val in pred['keypoint_annotations'].items():
            val = np.asarray(val).reshape(-1, 3)
            val = val[point_4_id]
            val = list(val.reshape(-1,))
            aa[key] = val

        predictions['annos'][pred['image_id']]['keypoint_annos'] = aa

    return predictions


def compute_oks(anno, predict, delta):
    """Compute oks matrix (size gtN*pN)."""

    anno_count = len(anno['keypoint_annos'].keys())
    predict_count = len(predict.keys())
    oks = np.zeros((anno_count, predict_count))
    if predict_count == 0:
        return oks.T

    # for every human keypoint annotation
    for i in range(anno_count):
        anno_key = list(anno['keypoint_annos'].keys())[i]
        anno_keypoints = np.reshape(anno['keypoint_annos'][anno_key], (14, 3))
        anno_keypoints = anno_keypoints[[0, 3, 12, 13], :]
        visible = anno_keypoints[:, 2] == 1
        bbox = anno['human_annos'][anno_key]
        scale = np.float32((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))
        if np.sum(visible) == 0:
            for j in range(predict_count):
                oks[i,  j] = 0
        else:
            # for every predicted human
            for j in range(predict_count):
                predict_key = list(predict.keys())[j]
                predict_keypoints = np.reshape(predict[predict_key], (4, 3))
                dis = np.sum((anno_keypoints[visible, :2]
                              - predict_keypoints[visible, :2]) ** 2, axis=1)
                oks[i, j] = np.mean(np.exp(-dis / 2 / delta[visible] ** 2 / (scale + 1)))
    return oks


def keypoint_eval(predictions, annotations, return_dict):
    """Evaluate predicted_file and return mAP."""

    oks_all = np.zeros((0))
    oks_num = 0

    # Construct set to speed up id searching.
    prediction_id_set = set(predictions['image_ids'])

    # for every annotation in our test/validation set
    for image_id in annotations['image_ids']:
        # if the image in the predictions, then compute oks
        if image_id in prediction_id_set:
            oks = compute_oks(anno=annotations['annos'][image_id],
                              predict=predictions['annos'][image_id]['keypoint_annos'],
                              delta=annotations['delta'])
            # view pairs with max OKSs as match ones, add to oks_all
            oks_all = np.concatenate((oks_all, np.max(oks, axis=1)), axis=0)
            # accumulate total num by max(gtN,pN)
            oks_num += np.max(oks.shape)
        else:
            # otherwise report warning
            return_dict['warning'].append(image_id + ' is not in the prediction JSON file.\n')
            # number of humen in ground truth annotations
            gt_n = len(annotations['annos'][image_id]['human_annos'].keys())
            # fill 0 in oks scores
            oks_all = np.concatenate((oks_all, np.zeros((gt_n))), axis=0)
            # accumulate total num by ground truth number
            oks_num += gt_n

    # compute mAP by APs under different oks thresholds
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold) / np.float32(oks_num))
    return_dict['score'] = np.mean(average_precision)

    return return_dict


def test_anno_file():
    """The evaluator."""
    gt_anno_file = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
    predicit_file = '/mnt/data/dataset/PoseData/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'

    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', help='prediction json file', type=str,
                        default='keypoint_predictions_example.json')
    parser.add_argument('--ref', help='annotation json file', type=str,
                        default='keypoint_annotations_example.json')
    args = parser.parse_args()

    # Initialize return_dict
    return_dict = dict()
    return_dict['error'] = None
    return_dict['warning'] = []
    return_dict['score'] = None

    # Load annotation JSON file
    start_time = time.time()
    annotations = load_annotations(anno_file=gt_anno_file,
                                   return_dict=return_dict)
    print
    'Complete reading annotation JSON file in %.2f seconds.' % (time.time() - start_time)

    # Load prediction JSON file
    start_time = time.time()
    predictions = load_predictions(prediction_file=predicit_file,
                                   return_dict=return_dict)
    print
    'Complete reading prediction JSON file in %.2f seconds.' % (time.time() - start_time)

    # Keypoint evaluation
    start_time = time.time()
    return_dict = keypoint_eval(predictions=predictions,
                                annotations=annotations,
                                return_dict=return_dict)
    print
    'Complete evaluation in %.2f seconds.' % (time.time() - start_time)

    # Print return_dict and final score
    pprint.pprint(return_dict)
    print
    'Score: ', '%.8f' % return_dict['score']


def test_net():
    from model.peleenet import PeleePoseNet
    from model.RTNet import RTNet, RTNet_Half
    import torch
    import cv2
    import os
    from pose_decode import decode_pose

    use_gpu = True
    is_resize = True
    four_out = False

    gt_anno_file = '/mnt/data/dataset/PoseData/ai_challenge_2017/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json'
    image_root_path = '/mnt/data/dataset/PoseData/ai_challenge_2017/ai_challenger_valid_test/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911'
    # model_file = 'result/checkpoint/1101/epoch_9_0.014181.cpkt'
    model_file = 'result/checkpoint/1226/epoch_22_0.005897.cpkt'
    # model_file = 'result/checkpoint/1217/epoch_6_0.008613.cpkt'
    # model_file = "result/checkpoint/1217/epoch_25_0.007546.cpkt"

    net = PeleePoseNet()
    # net = RTNet()
    # net = RTNet_Half(1)
    param = {'thre1': 0.1, 'thre2': 0.00, 'thre3': 0.5}

    if use_gpu:
        net = net.cuda()
    net.load_state_dict(torch.load(model_file))
    net.eval()

    image_list = []
    annos = json.load(open(gt_anno_file))
    for anno in annos:
        image_list.append(anno['image_id'])

    predictions = dict()
    predictions['image_ids'] = []
    predictions['annos'] = dict()

    for num, image_id in enumerate(image_list):

        # if num > 100:
            # break
        src_img = cv2.imread(os.path.join(image_root_path, image_id) + '.jpg')
        image = src_img.copy()
        if is_resize:
            image = cv2.resize(image, (368, 368))

        img = np.transpose(image, [2, 0, 1])
        img = np.asarray(img, dtype=np.float32) / 255
        img = torch.from_numpy(img)
        img = torch.Tensor.unsqueeze(img, 0)
        if use_gpu:
            img = img.cuda()
        if four_out:
            _, _, heatmaps, pafs = net(img)
        else:
            heatmaps, pafs = net(img)

        heatmaps = heatmaps.cpu().data.numpy().transpose(0, 2, 3, 1)
        pafs = pafs.cpu().data.numpy().transpose(0, 2, 3, 1)
        canvas, joint_list, person_to_joint_assoc = decode_pose(image, param, heatmaps[0], pafs[0])

        print('[{}] current Image:{} joins:{} persons:{}'.format(num, image_id, joint_list.size, person_to_joint_assoc.size))


        predictions['image_ids'].append(image_id)
        predictions['annos'][image_id] = dict()

        if joint_list.size < 1:
            predictions['annos'][image_id]['keypoint_annos'] = {}
            continue
        scale_x = src_img.shape[1] / image.shape[1]
        scale_y = src_img.shape[0] / image.shape[0]
        points = joint_list[:, 0:2] * [scale_x, scale_y]
        # points = joint_list[:, 0:2]

        person = {}
        for i, p in enumerate(person_to_joint_assoc):
            person['human%d'%i] = []
            for j in p[0:4]:
                if j < 0:
                    person['human%d' % i] += [0, 0, 1]
                else:
                    person['human%d' % i] += list(points[int(j)]) + [1]
        predictions['annos'][image_id]['keypoint_annos'] = person

    # with open('test111.json', 'w') as f:
    #     json.dump(predictions, f)

    return_dict = {}
    return_dict['error'] = None
    return_dict['warning'] = []
    return_dict['score'] = None
    annotations = load_annotations(gt_anno_file, return_dict )

    return_dict = keypoint_eval(predictions, annotations, return_dict)
    print(len(return_dict['warning']))
    print('error: ', return_dict['error'])
    print('score: ', return_dict['score'])


if __name__ == "__main__":
    test_net()
    # test_anno_file()

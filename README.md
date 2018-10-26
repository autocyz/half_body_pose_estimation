# Half Body Pose Estimation 
![](data/01d60fd2ef1f95c4f7dd414d306146f1268940fa_1.jpg)

## Introduction
This is a half body pose estimation project, which can detect four joints(right/left shoulder, neck, head) and three limbs(neck->left shoulder, neck->right shoulder, neck->head).

The algorithm adopted in this project is ``Realtime Multi-person 2D Pose Estimation using Part Affinity Fields ``.

## Dependencies
pytorch=0.4.1

tensorboadX=1.4

numpy=1.14.3

## Files
> model: network(CPM and PAF model ) 

> train: train code 

> inference: inference code 

> dataset: create input image and labels from AIChanllenge

> pose_decode: compute and plot joints and limbs from net output
 
 
 
 
 [1] Realtime Multi-person 2D Pose Estimation using Part Affinity
  
 [2] https://challenger.ai/dataset/keypoint 

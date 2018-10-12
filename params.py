

params_transform = dict()

# dataset or data info
params_transform['num_keypoint'] = 4
params_transform['paf_width_thre'] = 1
params_transform['crop_size_x'] = 368
params_transform['crop_size_y'] = 368
params_transform['feature_map_ratio'] = 8  # feature_map size = crop_size / feature_map_ratio

# data augmentation params
params_transform['scale_min'] = 0.5
params_transform['scale_max'] = 1.1
params_transform['scale_prob'] = 1
params_transform['target_dist'] = 0.6
params_transform['max_rotate_degree'] = 40
params_transform['center_perterb_max'] = 40
params_transform['flip_prob'] = 0.5
params_transform['np'] = 56
params_transform['sigma'] = 2.0

# train params
params_transform['epoch_num'] = 100
params_transform['batch_size'] = 16
params_transform['num_workers'] = 4
params_transform['learning_rate'] = 1e-4
params_transform['momentum'] = 0.9
params_transform['weight_decay'] = 1e-4
params_transform['nesterov'] = True

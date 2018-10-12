import os


def save_params(path, name, params):
    with open(os.path.join(path, name+'.txt'), 'w') as f:
        for key, val in params.items():
            f.write(key + ': ')
            f.write(str(val))
            f.write('\n')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    from params import params_transform
    save_params('./', 'params_transform', params_transform)
import numpy as np
import random
import os
import os.path as osp

import sys
import argparse

max_num = 5000

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Pre-process the classification dataset')
    parser.add_argument('--dataset', dest='dataset',
                        help='Dataset to be preprocessed',
                        default=None, type=str)
    parser.add_argument('--ratio', dest='ratio',
                        help='Ratio of the training data',
                        default=0.8, type=float)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def flush_dataset(dataset):
    with open(osp.join(dataset, 'list.txt'), 'rb') as f:
        dirs = [_dir.strip() for _dir in f.readlines() if _dir.strip('\n') != '']

    clses = []
    train_roidb = []
    test_roidb = []
    for this_dir in dirs:
        # skip list.txt
        if this_dir == 'list.txt':
            continue
        sub_clses, this_train_roidb, this_test_roidb = create_train_test(osp.join(dataset, this_dir), this_dir)
        clses += sub_clses
        train_roidb += this_train_roidb
        test_roidb += this_test_roidb

    with open(osp.join(dataset, 'sysnets.txt'), 'wb') as f:
        for ix, cls in enumerate(clses):
            f.write('{} {}\n'.format(cls, ix))
    with open(osp.join(dataset, 'train.txt'), 'wb') as f:
        for img in train_roidb:
            f.write('{} {}\n'.format(img[0], clses.index(img[1])))
    with open(osp.join(dataset, 'test.txt'), 'wb') as f:
        for img in test_roidb:
            f.write('{} {}\n'.format(img[0], clses.index(img[1])))

    return clses


def create_train_test(dataset, root):
    sub_clses = os.listdir(dataset)

    this_train_roidb = []
    this_test_roidb = []
    
    for sub_cls in sub_clses:
        num_total_roidb = 0
        _fs = os.listdir(osp.join(dataset, sub_cls))
        if os.path.isdir(osp.join(dataset, sub_cls, _fs[0])):
            for this_dir in _fs:
                sub_dir = osp.join(dataset, sub_cls, this_dir)
                sub_roidb = [(osp.join(root, sub_cls, this_dir, img), sub_cls) for img in os.listdir(sub_dir)]
                num_sub_roidb = max_num if len(sub_roidb)>max_num else len(sub_roidb)
                num_total_roidb += num_sub_roidb
                num_train = int(num_sub_roidb * args.ratio)
                # shuffle roidb
                random.shuffle(sub_roidb)
                this_train_roidb.extend(sub_roidb[: num_train])
                this_test_roidb.extend(sub_roidb[num_train:])
        else:
            sub_roidb = [(osp.join(root, sub_cls, img), sub_cls) for img in _fs]
            # shuffle roidb
            random.shuffle(sub_roidb)
            num_sub_roidb = max_num if len(sub_roidb)>max_num else len(sub_roidb)
            num_total_roidb += num_sub_roidb
            num_train = int(num_sub_roidb * args.ratio)
            this_train_roidb.extend(sub_roidb[: num_train])
            this_test_roidb.extend(sub_roidb[num_train: max_num])
        print 'Class {}: {} entris.'.format(sub_cls, num_total_roidb)

    return sub_clses, this_train_roidb, this_test_roidb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    dataset = osp.join('data', args.dataset)
    assert os.path.exists(dataset), '{} does not exist'.format(dataset)

    flush_dataset(dataset)

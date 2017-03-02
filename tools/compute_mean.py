import _init_paths
import numpy as np
import cv2
import argparse
import sys
import os.path as osp
from PIL import Image

from utils.timer import Timer

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Compute the channel mean of the dataset')
    parser.add_argument('-s', '--size', dest='input_size',
                        help='input image size', default=224, type=int)
    parser.add_argument('-d', '--dataset', dest='dataset',
                        help='dataset', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_img_list(base_dir):
    list_file = osp.join(base_dir, 'train.txt')
    with open(list_file) as f:
        img_list = [osp.join(base_dir, line.strip().split()[0]) for line in f if not line.strip() == '']
    return img_list


def comput_img_mean(img_list, img_size):
    timer = Timer()
    mean_sum = np.array([[[0., 0., 0.]]], dtype=np.float)
    for ix, img_file in enumerate(img_list):
        if (ix + 1) % 1000 == 0:
            print 'Processed {} files, average speed: {}'.format(ix + 1, timer.average_time)

        timer.tic()
        im = cv2.imread(img_file)
        # In case that opencv cannot support the image
        if im is None:
            print img_file
            Image.open(img_file).convert('RGB').save(img_file, 'jpeg')
            im = cv2.imread(img_file)
            print 'Convert {} to RGB jpeg file'.format(img_file) 
        im_size_x = im.shape[1]
        im_size_y = im.shape[0]
        im_scale_x = float(img_size) / float(im_size_x)
        im_scale_y = float(img_size) / float(im_size_y)
        # Resize the image
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        this_img_mean = np.mean(im, axis=(0, 1))
        mean_sum += this_img_mean
        timer.toc()

    mean_array = mean_sum / len(img_list)

    return mean_array


if __name__ == '__main__':
    args = parse_args()

    base_dir = osp.join('data', args.dataset)

    img_list = get_img_list(base_dir)
    mean_array = comput_img_mean(img_list, args.input_size)
    print 'mean:', mean_array
    np.save(osp.join(base_dir, 'mean.npy'), mean_array)
    print 'Save to {}'.format(osp.join(base_dir, 'mean.npy'))

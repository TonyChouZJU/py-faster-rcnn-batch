from tools.train_net import train
import os
import numpy as np
import caffe
from datafactory.imdb import IMDB
from datafactory.load import load_data

from configuration.config import def_cfg, cfg

import sys
import argparse


def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--size', dest='size',
                        help='image input size',
                        default='224', type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='fruits', type=str)
    parser.add_argument('--out', dest='out',
                        help='models to save in',
                        default='out', type=str)
    parser.add_argument('--batchsize', dest='batchsize',
                        help='Batch size',
                        default='64', type=str)
    parser.add_argument('--cfg', dest='cfg',
                        help='Configuration file',
                        default='GCLS', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    def_cfg('GCLS')

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    # setup the dataset's path
    dataset = os.path.join(cfg.TRAIN.DEFAULT_DATA_DIR, args.imdb_name)
    # load pixel mean
    pixel_means = None
    if os.path.exists(os.path.join(dataset, 'mean.npy')):
        pixel_means = np.load(os.path.join(dataset, 'mean.npy'))
        print 'Loaded mean.npy: {}'.format(pixel_means)
    else:
        print 'Cannot find mean.npy and we will use default mean.'
    # configure input size
    cfg.TRAIN.INPUT_SIZE = int(args.size)
    # set the batch size
    cfg.TRAIN.IMS_PER_BATCH = int(args.batchsize)

    imdb = IMDB()
    imdb.get_roidb(load_data, dataset=dataset)
    roidb = imdb.roidb

    if args.pretrained_model is None:
        args.pretrained_model = cfg.TRAIN.DEFAULT_PRETRAINED_MODEL

    train(args.max_iters, args.solver, roidb, args.out,
          pretrained_model=args.pretrained_model,
          pixel_means=pixel_means)

import _init_paths
from tools.train_net import train
import os
import numpy as np
import caffe
from datafactory.imdb import IMDB
from datafactory.load import load_data_with_boxes

from tools.train_net_with_boxes import get_training_roidb, train_net
import fast_rcnn.config as fconfig
from fast_rcnn.config import cfg_from_file
from configuration.config import def_cfg

import sys
import pprint

gpu_id = 0
solver = '/home/zyb/VirtualDisk500/exhdd/Recognition-master/models/UniqueDET/solver_debug.prototxt'
max_iters = 100000
size = 224
imdb_name = 'UniqueDET'
out = 'out'
cfg = '/home/zyb/VirtualDisk500/exhdd/Recognition-master/experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = '/home/zyb/VirtualDisk500/exhdd/Recognition-master/pretrained_models/VGG_CNN_M_1024.v2.caffemodel'

if __name__ == '__main__':

    def_cfg('UDET')

    cfg_from_file(cfg)
    pprint.pprint(fconfig.cfg)

    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    # setup the dataset's path
    dataset = os.path.join('..', 'data', imdb_name)
    # load pixel mean
    pixel_means = None
    if os.path.exists(os.path.join(dataset, 'mean.npy')):
        pixel_means = np.load(os.path.join(dataset, 'mean.npy'))
        fconfig.cfg.PIXEL_MEANS = pixel_means
        print 'Loaded mean.npy: {}'.format(pixel_means)
    else:
        print 'Cannot find mean.npy and we will use default mean.'

    imdb = IMDB()
    imdb.get_roidb(load_data_with_boxes, dataset=dataset)
    roidb = get_training_roidb(imdb)

    np.random.seed(fconfig.cfg.RNG_SEED)
    caffe.set_random_seed(fconfig.cfg.RNG_SEED)

    train_net(solver, roidb, out,
              pretrained_model=pretrained_model, max_iters=max_iters)
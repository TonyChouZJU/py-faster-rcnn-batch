from tools.train_net import train
import os
import os.path as osp
import numpy as np
import caffe
from datafactory.imdb import IMDB
from datafactory.load_json import load_data_with_boxes, load_data

from tools.train_net_with_boxes import get_training_roidb, train_net
import fast_rcnn.config as fconfig
from fast_rcnn.config import cfg_from_file
from configuration.config import def_cfg

import sys
import argparse
import pprint

from tools.wl_parser import wl_parser
from tools.preprocess_json import save_list_file, compute_img_mean, create_train_test

import shutil

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--config_file', dest='config_file',
                        help='config json file to train this model',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    
    wl_args = wl_parser(args.config_file)
    resource = wl_args.getParam('resource')
    isDET = int(wl_args.getParam('DLType'))
    solver_proto = wl_args.getParam('solver_proto').encode("utf-8")
    train_proto = wl_args.getParam('train_proto').encode("utf-8")
    test_proto = wl_args.getParam('test_proto').encode("utf-8")
    cfg_file = wl_args.getParam('cfg_yml').encode("utf-8")
    pretrained_model = wl_args.getParam('pretrained_model').encode("utf-8")
    output_dir = wl_args.getParam('output').encode("utf-8")
    sysnets_path = wl_args.getParam('sysnets').encode("utf-8")
    gpu_id =  int(wl_args.getParam('gpu_id'))
    iters = int(wl_args.getParam('iters'))
    train_img_size = int(wl_args.getParam('img_size'))
    batch_size = int(wl_args.getParam('batch_size'))
    mean_file = wl_args.getParam('mean_file')

    dataset = '/'
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    if isDET != 0:
        def_cfg('UDET')
        cfg_from_file(cfg_file)
    else:
        def_cfg('CLS')
        cfg.TRAIN_INPUT_SIZE = int(train_img_size)
        cfg.TRAIN_IMS_PER_BATCH = int(batch_size)
    pprint.pprint(fconfig.cfg)
    
    #change the first line of solver_proto with train_proto
    f_solver = open(solver_proto,'r')
    solver_lines = f_solver.readlines()
    solver_lines[0] = 'train_net: ' + '\"{}\"\n'.format(train_proto)
    with open(solver_proto,'w+') as f_solver_new:
	f_solver_new.writelines(solver_lines)
    # setup the dataset's path
    save_list_file(resource,output_dir,isDET)
    #save a copy of sysnets to output dir
    if not os.path.exists(osp.join(output_dir,'sysnets.txt')):
        shutil.copy(sysnets_path,osp.join(output_dir,'sysnets.txt'))
    # load pixel mean
    if mean_file == None:
        with open(osp.join(output_dir, 'train.txt')) as _f_train:
            img_list = [ line.strip().split()[0] + '.jpg' for line in _f_train if not line.strip()=='']
        mean_array = compute_img_mean(img_list,train_img_size)
        print 'mean:',mean_array
        np.save(osp.join(output_dir,'mean.npy'), mean_array)
        print 'Save to {}'.format(osp.join(output_dir,'mean.npy'))
    pixel_means = None
    if os.path.exists(os.path.join(output_dir, 'mean.npy')):
        pixel_means = np.load(os.path.join(output_dir, 'mean.npy'))
        fconfig.cfg.PIXEL_MEANS = pixel_means
        print 'Loaded mean.npy: {}'.format(pixel_means)
    else:
        print 'Cannot find mean.npy and we will use default mean.'
    np.random.seed(fconfig.cfg.RNG_SEED)
    caffe.set_random_seed(fconfig.cfg.RNG_SEED)
    imdb = IMDB()
    if isDET:
        imdb.get_roidb(load_data_with_boxes, dataset=output_dir)
        roidb = get_training_roidb(imdb)
	#TODO pixel_means
        train_net(solver_proto, roidb, output_dir, pretrained_model=pretrained_model, max_iters=iters)
    else:
        imdb.get_roidb(load_data, dataset=output_dir)
        roidb = imdb.roidb
        train(solver_proto, roidb, output_dir, pretrained_model=pretrained_model, max_iters=iters, pixel_means=pixel_means)

import _init_paths
import numpy as np
import cv2
import argparse
import sys
import os.path as osp
from PIL import Image
import random

from utils.timer import Timer

from contextlib import nested

def create_train_test(all_list, train_list_ratio = 0.8):
    #all_list = np.random.permutation(all_list_2)
    random.shuffle(all_list)
    num_list = len(all_list)
    num_train_list = int(num_list * train_list_ratio)
    num_test_list = num_list - num_train_list
    train_list = all_list[:num_train_list]
    test_list = all_list[num_train_list:]
    return train_list, test_list
    
def compute_img_mean(img_list, img_size):
    timer = Timer()
    mean_sum = np.array([[[0., 0., 0.]]], dtype=np.float)
    for ix, img_file in enumerate(img_list):
        if (ix + 1) % 1000 == 0:
            print 'Processed {} files, average speed: {}'.format(ix + 1, timer.average_time)

        timer.tic()
        im = cv2.imread(img_file)
        # In case that opencv cannot support the image
        if im is None:
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

def save_list_file(resources_list, dataset, isDET):
    # DL type: DET
    # The element of resources_list should be a string
    if isDET == 1:        
        assert isinstance(resources_list,list),'DL type is DET,the resources_list should be a list'
        train_xml_list, test_xml_list = create_train_test(resources_list, train_list_ratio = 0.8)
	

        with open(osp.join(dataset,'train.txt'),'wb') as f_train:
            for _xml_file_train in train_xml_list:
		_xml_file_train = str(_xml_file_train)
                f_train.write(_xml_file_train.split('.')[0] + '\n')
        with open(osp.join(dataset,'test.txt'),'wb') as f_test:
            for _xml_file_test in test_xml_list:
		_xml_file_test = str(_xml_file_test)
                f_test.write(_xml_file_test.split('.')[0] + '\n')
    #DL type: CLS
    else:
        assert isinstance(resources_list,list),'DL type is CLS, the resources_list should be a list list'
        with nested(open(osp.join(dataset,'train.txt'),'w+') , open(osp.join(dataset,'text.txt'),'w+')) as (f_train,f_test):
            for ix, this_list in resources_list:
                assert isinstance(this_list[0],str),'the element of sublist {} should be a string'.format(ix)
                this_train_list, this_test_list = create_train_test(this_list, train_list_ratio = 0.8)
                for _jpg_file in this_train_list:
		    _jpg_file = _jpg_file.encode('utf-8')
                    f_train.write('{} {}'.format(_jpg_file.split('.')[0],ix))
                for _jpg_file in this_test_list:
		    _jpg_file = _jpg_file.encode('utf-8')
                    f_test.write('{} {}'.format(_jpg_file.split('.')[0], ix))

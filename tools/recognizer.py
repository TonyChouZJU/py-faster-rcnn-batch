__author__ = 'wonderland'
__version__ = '0.0'

"""
Recognizer is an image recognizer(classifier or detector) specialization of Net

result format
test_type=0 and is_cascade == False:
        input:img_path='/tmp/test.jpg', rois=[[0,0,w,h]]
        output:  OrderedDict
                    {'apple':[[0,0,w,h,0.8]],
                     'orange':[[0,0,w,h,0.1]],
                     'lemon':[[0,0,w,h,0.1]],
                     'pear':[[0,0,w,h,0.0]],
                     'peach':[[0,0,w,h,0.0]]
                    }
test_type=0 and is_cascade == True:
suppose this is the cascade classify for 'car'
        input:img_path='/tmp/car.jpg', rois=[[x1,y1,w1,h1,0.8],[x2,y2,w2,h2,0.9],...,[xN,yN,wN,hN,1]]
        output:    list of OrderedDict
        first item:  {'audi A4':[[x1,y1,w1,h1,0.8]],
                     'audi A3':[[x1,y1,w1,h1,0.1]],
                     'bmw x3':[[x1,y1,w1,h1,0.1]],
                     'bmw x1':[[x1,y1,w1,h1,0.0]],
                     'audi q3':[[x1,y1,w1,h1,0.0]]
                    }
test_type=1:
        input:img_path='/tmp/test.jpg', rois=[[0,0,w,h]]
        output:     {'car':[[x11,y11,w11,h11,0.8],[x12,y12,w12,h12,0.9],...,[x1N,y1N,w1N,h1N,1]],
                     'bus':[[x21,y21,w21,h21,0.9],[x22,y22,w22,h22,0.1],...,[x2N,y2N,w2N,h2N,0.8]],
                        ...,
                     'train':[[xN1,yN1,wN1,hN1,0.0]]
                    }
"""
import _init_paths
import numpy as np
import caffe
import cv2
import os
import collections
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg

cfg.TEST.HAS_RPN = True

class Recognizer(caffe.Net):
    """
    Recognizer extends Net for image recognition
    using classifier or detector
    """
    def __init__(self, model_def_file, pretrained_model_file, test_type, is_cascade, mean_file, gpu_mode, gpu_id, label_file, image_dim=255, nms_thresh=0.3, conf_thresh=0.8, raw_scale=256):
        if not os.path.isfile(label_file):
            raise IOError(('{:s} not found.\nDid you feed a valid label file path').format(label_file))

        if gpu_mode:
            caffe.set_mode_gpu()
            if gpu_id is not None:
                caffe.set_device(gpu_id)
            else:
                caffe.set_device(0)
        else:
            caffe.set_mode_cpu()
        self.test_type = test_type
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.is_cascade = is_cascade
        #preprocess configuration
        assert type(image_dim) == int, 'image_dim type should be int'
        mean_array = np.load(mean_file)
        if len(mean_array) == 3:
            mean_array = mean_array.mean(1).mean(1)
        elif len(mean_array) == 1:
            mean_array = mean_array[0][0]
        else:
            mean_array = np.array([])
        self.class_labels, self.class_indexes = self.load_label_file(label_file)

        #test_type ==0 refers to classification
        if self.test_type == 0:
            self.net = caffe.Classifier(model_def_file, pretrained_model_file, image_dims=(image_dim,image_dim), mean=mean_array, channel_swap=(2,1,0), raw_scale=raw_scale)
        else:
            self.net = caffe.Net(model_def_file, pretrained_model_file, caffe.TEST)


    def load_label_file(self,label_file):
        class_labels = {}
        class_indexes = {}
        with open(label_file,'rb') as sysnets:
            clabels = [clabel.strip('\n') for clabel in sysnets.readlines()]
            for clabel in clabels:
                clabel = clabel.split()
                class_labels[int(clabel[1])] = clabel[0]
                class_indexes[clabel[0]] = int(clabel[1])
        return class_labels, class_indexes

    def classify_cascade(self, imgs_path, rois=None, oversample=True):
        """
        classify the rois of detection results
        :param imgs_path: path of imgs.
            while images are iterable of (H x W x K) input ndarrays.
        :param rois: numpy array
            sample [[x1,y1,w1,h1,score1],
                    [x2,y2,w2,h2,score2],
                    ...
                    [xN,yN,wN,hN,scoreN]]
        :param oversample: boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only predection when False.
        :return: results
            list of dict.
        """

        assert self.test_type == 0, 'classify can only be used when test_type is 0'
        assert len(imgs_path)==1, 'classify_single can only used for single image list'
        results = [None] * len(rois)
        try:
            input_image = caffe.io.load_image(imgs_path[0])
        except Exception as err:
            print ('Uploaded image open error: %s', err)
            return results
        img_H = input_image.shape[0]
        img_W = input_image.shape[1]
        for idx, roi in enumerate(rois):
            x, y, w, h = roi[:4]
            if not (x >= 0 and y >= 0 and x+w < img_W and y+h < img_H):
                continue
            try:
                result = self.classify_raw(input_image[y:y+h,x:x+w,:],oversample)
            except Exception as err:
                continue
            results[idx] = result
        return results

    def classify(self, imgs_path, oversample=True):
        """
        Predict classification probabilities of inputs.

        Parameters
        ----------
        imgs_path: path of imgs.
            while images are iterable of (H x W x K) input ndarrays.
        oversample: boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only predection when False.
        Returns:
        --------
        predictions: orderedDict. Key: label_name, val: (1 x 5) ndarray of class probabilities for this image
        """

        assert self.test_type == 0, 'classify can only be used when test_type is 0'
        try:
            input_image = caffe.io.load_image(imgs_path)
        except Exception as err:
            print ('Uploaded image open error: %s', err)
            return None
        result = self.classify_raw(input_image,oversample)
        return result

    def classify_raw(self, raw_img, oversample=True):
        scores = self.net.predict([raw_img], oversample=oversample).flatten()
        indices_top5 = (-scores).argsort()[:5]
        scores_top5 = scores[indices_top5]
        result = collections.OrderedDict()
        for idx, cls in enumerate(indices_top5):
            result[self.class_labels[cls]] = \
                np.hstack((np.array([0, 0, raw_img.shape[1], raw_img.shape[0]]), scores_top5[idx]))[np.newaxis, ]
        return result

    # TODO:classifiy multi rois using classify_batches

    def detect(self, imgs_path, rois=None):
        """
        :param imgs_path:
        :param rois: default is None
        :return: dict() which key is label_name and value is ndarray (nums_proposals, 4+1)
                sample: {'n0000_car':[[x1,y1,w1,h1,score1],
                                        [x2,y2,w2,h2,score2],
                                        ...
                                        [xN,yN,wN,hN,scoreN]],
                    'n0001_plane':[[x1,y1,w1,h1,score1],
                                    [x2,y2,w2,h2,score2],
                                    ...
                                    [xN,yN,wN,hN,scoreN]]
                    }
        """
        assert self.test_type == 1, 'classify can only be used when test_type is 0'
        CLASSES = [self.class_labels[i] for i in range(len(self.class_labels))]
        results = None
        try:
            im = cv2.imread(imgs_path)
        except Exception as err:
            print 'cannot loading image'
            return results
        results = self.detect_raw(im, CLASSES, self.nms_thresh, self.conf_thresh )
        return results

    def detect_raw(self, raw_img, CLASSES, NMS_THRESH=0.3, CONF_THRESH=0.8):
        """
        :param raw_img: (H x W x K) input ndarrays, color channel should be BGR (cv2 default)
        :param CLASSES: classes list
        :param NMS_THRESH: non-maximum suppression ratio
        :param CONF_THRESH: score confidence ratio
        :return: dict() which key is label_name and value is ndarray (nums_proposals, 4+1)
                sample: {'n0000_car':[[x1,y1,w1,h1,score1],
                    [x2,y2,w2,h2,score2],
                    ...
                    [xN,yN,wN,hN,scoreN]],
                    'n0001_plane':[[x1,y1,w1,h1,score1],
                    [x2,y2,w2,h2,score2],
                    ...
                    [xN,yN,wN,hN,scoreN]]}
        """
        # Detect all object classes and regress object bounds
        # scores is numpy array which shape is (nums_proposals , num_classes+1)
        # boxes is numpy array which shape is (nums_proposals , 4*(num_classes+1) )
        scores, boxes = im_detect(self.net, raw_img)
        results = dict()
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            dets = dets[inds, :]
            sorted_idx = np.argsort(dets,axis=0)[:, 4][::-1]
            dets = dets[sorted_idx,:]
            results[cls] = dets
        return results

    def predict(self, imgs_path, rois):
        if self.test_type ==1:
            return self.detect(imgs_path, rois)
        elif self.test_type == 0 and not self.is_cascade:
            return self.classify(imgs_path, rois)
        elif self.test_type == 0 and self.is_cascade:
            return self.detect(imgs_path, rois)






        

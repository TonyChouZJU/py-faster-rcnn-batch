# --------------------------------------------------------
# Tools for loading datasets
# Written by Stephen
# --------------------------------------------------------

import os
import os.path as osp

import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse


def load_images(dataset, fname='train.txt'):
    with open(osp.join(dataset, fname), 'rb') as f:
        train_datas = [train_data.strip('\n') for train_data in f.readlines()]
    roidb = []
    for train_data in train_datas:
        train_data = train_data.split()
        roidb.append({
            'im_path': train_data[0],
            'label': int(train_data[1]),
            'flipped': False,
            'transposed': False
        })
    return roidb


def load_xml_annotation(num_classes, xml, class_indexes):
    tree = ET.parse(xml)
    objs = tree.findall('object')
    num_objs = len(objs)

    # boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    # gt_classes = np.zeros((num_objs), dtype=np.int32)
    # overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
    # # "Seg" area for pascal is just the box area
    # seg_areas = np.zeros((num_objs), dtype=np.float32)

    boxes = []
    gt_classes = []
    overlaps = []
    # "Seg" area for pascal is just the box area
    seg_areas = []

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        objname = obj.find('name').text.lower().strip()
        if objname not in class_indexes:
            continue
        else:
            cls = class_indexes[objname]
            boxes.append([x1, y1, x2, y2])
            gt_classes.append(cls)
            this_overlaps = [0] * num_classes
            this_overlaps[cls] = 1.0
            overlaps.append(this_overlaps)
            seg_areas.append((x2 - x1 + 1) * (y2 - y1 + 1))

    overlaps = np.array(overlaps, dtype=np.float32)

    return {'boxes': np.array(boxes, dtype=np.uint16),
            'gt_classes': np.array(gt_classes, dtype=np.int32),
            'gt_overlaps': scipy.sparse.csr_matrix(overlaps),
            'flipped': False,
            'transposed': False,
            'seg_areas': np.array(seg_areas, dtype=np.float32)}


def load_images_with_boxes(dataset, class_indexes, fname='train.txt'):
    with open(osp.join(dataset, fname), 'rb') as f:
        train_datas = [train_data.strip('\n') for train_data in f.readlines()]
    roidb = []
    num_classes = len(class_indexes)
    for train_data in train_datas:
        #im_path = osp.join(dataset, 'Data', train_data + '.jpg')
        #xml = osp.join(dataset, 'Annotations', train_data + '.xml')
        im_path = train_data + '.jpg'
        xml =  train_data + '.xml'
        entry = load_xml_annotation(num_classes, xml, class_indexes)
        entry['im_path'] = im_path
        if not len(entry['boxes']) == 0:
            roidb.append(entry)
    return roidb


def load_sysnets(dataset):
    sysnets = open(osp.join(dataset, 'sysnets.txt'), 'rb')
    clabels = [clabel.strip('\n') for clabel in sysnets.readlines()]
    class_labels = {}
    class_indexes = {}
    for clabel in clabels:
        clabel = clabel.split()
        class_labels[int(clabel[1])] = clabel[0]
        class_indexes[clabel[0]] = int(clabel[1])
    return class_labels, class_indexes


def load_data(kargws):
    """ class_labels is a dictionary and roidb is a list of entries,
    which is in the form of {'im_path':..., 'label'..., 'flipped':..., 'transposed':...}
    """
    dataset = kargws['dataset']
    clabels, cindexes = load_sysnets(dataset)
    roidb = load_images(dataset)
    return clabels, roidb


def load_data_with_boxes(kargws):
    """ class_labels is a dictionary and roidb is a list of entries,
        which is in the form of {'im_path':..., 'label'..., 'flipped':..., 'transposed':...}
        """
    dataset = kargws['dataset']
    clabels, cindexes = load_sysnets(dataset)
    roidb = load_images_with_boxes(dataset, cindexes)
    return clabels, roidb

import numpy.random as npr
import xml.etree.ElementTree as ET
import os
import os.path as osp
import sys
import argparse


def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Pre-process the detection dataset')
    parser.add_argument('--dataset', dest='dataset',
                        help='Dataset to be preprocessed',
                        default=None, type=str)
    parser.add_argument('--ratio', dest='ratio',
                        help='Ratio of the training data',
                        default=0.8, type=float)
    parser.add_argument('--lo', dest='lo',
                        help='logo or obj or all',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def flush_dataset(dataset, ratio, dtype):
    with open(osp.join(dataset, 'list.txt'), 'rb') as f:
        dirs = [_dir.strip() for _dir in f.readlines() if _dir.strip('\n') != '']

    # First step: load all xml files but split postfix
    files = []
    for _dir in dirs:
        _dir_root = osp.join(dataset, 'Annotations', _dir)
        sub_dirs = os.listdir(_dir_root)
        for sub_dir in sub_dirs:
            this_xml_names = os.listdir(osp.join(_dir_root, sub_dir))
            # split postfix
            this_files = [ osp.join(_dir, sub_dir, osp.splitext(xml_name)[0]) for xml_name in this_xml_names]
            files += this_files

    # Second step: build sysnets.txt
    sysnets = osp.join(dataset, 'sysnets.txt')
    if not osp.exists(sysnets):
        sysnets = open(sysnets, 'wb')
        classes = []
        for _file in files:
            xml = osp.join(dataset, 'Annotations', _file + '.xml')
            tree = ET.parse(xml)
            objs = tree.findall('object')

            for ix, obj in enumerate(objs):
                objname = obj.find('name').text.lower().strip()
                if objname not in classes:
		    print 'xml:',xml
                    if dtype == 'obj' and objname[:3] == 'obj':
                        classes.append(objname)
                    elif dtype == 'logo' and objname[:4] == 'logo':
                        classes.append(objname)
                    elif dtype == 'all':
                        classes.append(objname)
        classes.sort()
        # insert __background__
        classes.insert(0, '__background__')
        print 'Classes: ', classes
        for ind, _class in enumerate(classes):
            sysnets.write(_class + ' ' + str(ind) + '\n')
        sysnets.close()
    else:
        print 'sysnets.txt exists and skip building sysnets.txt'

    # Third step: randomly permute dataset and create train.txt and test.txt
    num_files = len(files)
    num_train = int(ratio * num_files)
    files = npr.permutation(files)
    train_files = files[: num_train]
    test_files = files[num_train: ]
    with open(osp.join(dataset, 'train.txt'), 'wb') as f:
        for _file in train_files:
            f.write(_file + '\n')
    with open(osp.join(dataset, 'test.txt'), 'wb') as f:
        for _file in test_files:
            f.write(_file + '\n')

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    dataset = osp.join('data', args.dataset)
    assert os.path.exists(dataset), '{} does not exist'.format(dataset)
    assert args.lo is not None, 'Obj or Logo is not defined'

    flush_dataset(dataset, ratio=args.ratio, dtype=args.lo)

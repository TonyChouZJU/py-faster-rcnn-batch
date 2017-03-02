import cv2
import numpy as np
from PIL import Image

from configuration.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


def get_image_blob(db, pixel_means, itype):
    """ If the image is for training, then we consider flipped and transpoed,
     otherwise, we doesn't consider these features
    """
    processed_ims = []

    for data in db:
        # First step: load the image
        im_path = data['im_path']
        img = cv2.imread(im_path)
        # In case that opencv cannot support the image's format
        if img is None:
            out = im_path[: im_path.find('.')] + '.jpg'
            print '{} -> {}'.format(data['im_path'], out)
            Image.open(im_path).convert('RGB').save(out, 'jpeg')
            img = cv2.imread(out)

        # Second step: flip or transpose the image in training
        if itype == 'TRAIN':
            if data['flipped']:
                img = img[:, ::-1]
            if data['transposed']:
                img = img[::-1]

        img = img.astype(np.float32, copy=False)
        img -= pixel_means

        input_size = cfg.TRAIN.INPUT_SIZE if itype == 'TRAIN' else cfg.TEST.INPUT_SIZE
        img = prep_im_for_blob( img,
                                input_size,
                                input_size)

        processed_ims.append(img)

    blob = im_list_to_blob(processed_ims)

    return blob


def get_minibatch(db, num_classes, itype, pixel_means=None):
    num_images = len(db)

    if pixel_means is None:
        # It is a default pixel mean from py-faster-rcnn even though it may be not exact
        pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])

    im_blob = get_image_blob(db, pixel_means, itype)
    labels_blob = np.zeros((num_images, num_classes), dtype=np.float32)

    blobs = {
        'data': im_blob
    }

    for im_i in range(num_images):
        label = db[im_i]['label']
        labels_blob[im_i, label] = 1

    blobs['label'] = labels_blob

    return blobs


from configuration.config import cfg
import PIL.Image as Image

# Description
# There are two kinds of entries:
# For entry without 'bboxes':
#   {'im_path':..., 'label'..., 'flipped':..., 'transposed':...}
# For entry with 'bboxes':
#   {'im_path':..., 'label'..., 'flipped':..., 'transposed':...,
#   'gt_classes': ..., 'gt_overlaps': ..., 'seg_areas': ...}


class IMDB(object):
    """Image Database Class for image pre-processing."""
    def __init__(self):
        self.original_roidb = []
        self.extend_roidb = []
        self.class_labels = {}
        self._roidb_handle = None
        self.roidb = []
        # Used by faster rcnn roidb
        self.image_index = []
        self.num_images = 0

    def image_path_at(self, index):
        return self.roidb[index]['im_path']

    def append_flipped_image(self):
        """Append flipped image to the Region-of-Interest database."""
        for entry in self.original_roidb:
            flipped_entry = entry.copy()
            flipped_entry['flipped'] = True
            # change 'boxes' element if 'boxes' in entry
            if 'boxes' in entry:
                size = Image.open(entry['im_path']).size
                boxes = entry['boxes'].copy()
                oldx1 = boxes[:, 0].copy()
                oldx2 = boxes[:, 2].copy()
                boxes[:, 0] = size[0] - oldx2 - 1
                boxes[:, 2] = size[0] - oldx1 - 1
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                flipped_entry['boxes'] = boxes
            self.extend_roidb.append(flipped_entry)

    def append_transpose_image(self):
        """Append transpose image to the Region-of-Interest database."""
        for entry in self.original_roidb:
            transposed_entry = entry.copy()
            transposed_entry['transposed'] = True
            # change 'boxes' element if 'boxes' in entry
            if 'boxes' in entry:
                size = Image.open(entry['im_path']).size
                boxes = entry['boxes'].copy()
                oldy1 = boxes[:, 1].copy()
                oldy2 = boxes[:, 3].copy()
                boxes[:, 1] = size[1] - oldy2 - 1
                boxes[:, 3] = size[1] - oldy1 - 1
                assert (boxes[:, 2] >= boxes[:, 0]).all()
                transposed_entry['boxes'] = boxes
            self.extend_roidb.append(transposed_entry)

    def get_roidb(self, handle, **kwargs):
        """Get the Region-of-Interest generation function handle."""
        self._roidb_handle = handle

        self.class_labels, self.original_roidb = self._roidb_handle(kwargs)
        if cfg.TRAIN.USE_FLIPPED_IMAGE:
            self.append_flipped_image()
        if cfg.TRAIN.USE_TRANSPOSED_IMAGE:
            self.append_transpose_image()

        self.roidb = self.original_roidb + self.extend_roidb
        self.image_index = [i for i in range(len(self.roidb))]
        self.num_images = len(self.image_index)

        return self.roidb

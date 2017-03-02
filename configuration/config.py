# --------------------------------------------------------
# Default configuration file
# Written by Stephen
# --------------------------------------------------------

from easydict import EasyDict as edict
import numpy as np
import GeneralCLS_config
import UniqueDET_config

__C = edict()

cfg = __C

__C.TRAIN = edict()

__C.TEST = edict()

# Default training configuration for General Classification

__C.TRAIN.IMS_PER_BATCH = 1

__C.TRAIN.SCALES = (600,)

__C.TRAIN.HAS_RPN = True

__C.TRAIN.MAX_SIZE = 1000

__C.TRAIN.DEFAULT_DATA_DIR = 'data'

__C.TRAIN.SNAPSHOT_ITERS = 100000

__C.TRAIN.USE_FLIPPED_IMAGE = False

__C.TRAIN.USE_TRANSPOSED_IMAGE = False

__C.TRAIN.INPUT_SIZE = 224

__C.TEST.INPUT_SIZE = 384

cfgs = {
    'GCLS': GeneralCLS_config.cfg,
    'UDET': UniqueDET_config.cfg
}

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def def_cfg(ctype):
    assert ctype in cfgs, 'Unknown configurations {}'.format(ctype)
    _merge_a_into_b(cfgs[ctype], __C)

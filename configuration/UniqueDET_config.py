from easydict import EasyDict

__C = EasyDict()

cfg = __C

__C.TRAIN = EasyDict()

__C.TEST = EasyDict()

# Default training configuration for Unique Detection

__C.TRAIN.IMS_PER_BATCH = 1

__C.TRAIN.DEFAULT_DATA_DIR = 'data'

__C.TRAIN.SNAPSHOT_ITERS = 10000

__C.TRAIN.USE_FLIPPED_IMAGE = True

__C.TRAIN.USE_TRANSPOSED_IMAGE = False

__C.TRAIN.INPUT_SIZE = 224

__C.TEST.INPUT_SIZE = 224

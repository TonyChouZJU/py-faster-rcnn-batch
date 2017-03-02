import _init_paths
import os
from utils.timer import Timer

import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2

import configuration.config as ccfg


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process.
    """
    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None, pixel_means=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        if (not os.path.exists(output_dir)) or \
                (os.path.exists(output_dir) and not os.path.isdir(output_dir)):
            os.mkdir(output_dir)

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            self.solver.net.copy_from(pretrained_model)
            print 'Loaded pretrained model weights from {}.'.format(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)
        
        print 'Number of entries: {}'.format(len(roidb))
        self.solver.net.layers[0].set_roidb(roidb, pixel_means)

    def snapshot(self):
        """Take a snapshot of the network.
        This enables easy use at test-time.
        """
        net = self.solver.net

        filename = self.solver_param.snapshot_prefix + \
                   '_iter_{:d}'.format(self.solver.iter) + '.caffemodel'
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % ccfg.cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths


def train(max_iters, solver_prototxt,
          roidb, output_dir, pretrained_model=None, pixel_means=None):
    print 'Constructing Solver-Wrapper .........'
    solver = SolverWrapper(solver_prototxt, roidb,
                           output_dir, pretrained_model, pixel_means)
    print 'Size of roidb is {}'.format(len(roidb))
    print 'Begin training model ........'
    solver.train_model(max_iters)

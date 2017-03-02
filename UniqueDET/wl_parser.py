__author__ = 'wonderland'
__version__ = '0.0'
"""
 Interface for parsing computation json params_json_file
type: dict
key: 'info',
        'solver_proto',
        'train_proto',
        'test_proto',
        'cfg.yml',
        'pre_trained_model'
opt_key:'gpu_id',
        'iters',
        'size',
        'batch_size'
"""
import json
import numpy as np
import time
from collections import defaultdict

class wl_parser:
    def __init__(self, params_json_file=None):
        """
        Constructor of Wonderland DL class for processing parameter files
        :param param_json_file(str) : location of parameter file
        :return:
        """
        self.params_dict = defaultdict(str) 
        self.params_keys = ['info', 
            'solver_proto',
            'train_proto',
            'test_proto',
            'cfg_yml',
            'weights_caffemodel',
            'output_dir']
        self.params_opt_dict = {'gpu_id_int':0,
                'iters_int':800000,
                'img_size_int':224,
                'batchsize_int':64}
         
        if not param_json_file is None:
            print 'loading params into memory ...'
            tic = time.time()
            params = json.load(open(params_json_file),'r')
            assert type(params) == dict, "params_json_file format %s not supported" %(type(params))
            print 'Done (t= %0.2fs)'%(time.time()-tic)
            self.parse_params(params)
            

    def parse_params(self,params_):
        #parsing params from params_dict
        for param_key in self.params_keys:
            assert params_.has_key(param_key), "key %s missing!"%(param_key)
            self.params_dict[param_key] = params_[param_key]

        for param_opt_key in self.params_opt_dict.keys():
            if params_.has_key(param_opt_key):
                self.params_opt_dict[param_opt_key] = params_[param_opt_key]
            else:
                print param_opt_key,' using default value:',self.params_opt_dict[param_opt_key]
    
    def getParam(self, key_name):
        if self.params_dict.has(key_name):
            return self.params_dict[key_name]
        else:
            return self.params_opt_dict[key_name]
        
    def info(self):
        """
        Print information about the json file.
        :return:
        """
        for key,value in self.params_dict ['info'].items():
            print '%s: %s'%(key, value)

__author__ = 'wonderland'
__version__ = '0.0'
"""
 Interface for parsing computation json params_json_file
type: json
    "info":"{"time":time, {...},...}",
    "resource":
    [
        [apple_pic_path1,apple_pic_path2,apple_pic_path3],
        [banana_pic_path1,banana_pic_path2,banana_pic_path3],
        ...
    ],
    #"parameters":
   #{
        "DLType":bool
        "solver_proto":filepath,
        "train_proto":filepath,
        "test_proto":filepath,
        "cfg_yml":filepath, 
	"pretrained_model":filepath, 
	"output":dirpath,
        "sysnets":sysnet_path
    #},
    #"option":
    #{
        "gpu_id":id,
        "iters":iters,
        "mean_file":mean_file_path
        "img_size":size1,
        "batch_size":size2
    #}
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
        self.params_keys = ['resource',
            'info', 
	    'DLType',
            'solver_proto',
            'train_proto',
            'test_proto',
            'cfg_yml',
            'pretrained_model',
            'output',
            'sysnets']
        self.params_opt_dict = {'gpu_id':0,
                'iters':800000,
                'img_size':224,
                'batch_size':16,
                'mean_file':None}
         
        if not params_json_file is None:
            print 'loading params into memory ...'
            tic = time.time()
            json_file = open(params_json_file,'r')
            params = json.load(json_file,encoding = "utf-8")
            json_file.close()
            assert type(params) == dict, "params_json_file format %s not supported" %(type(params))
            print 'Done (t= %0.2fs)'%(time.time()-tic)
            self.parse_params(params)
            

    def parse_params(self,params_):
        #parsing params from params_dict
        for param_key in self.params_keys:
            assert params_.has_key(param_key), "key %s missing!"%(param_key)
            self.params_dict[param_key] = params_[param_key]

        for param_opt_key in self.params_opt_dict.keys():
            if params_.has_key(param_opt_key) and not params_[param_opt_key] is None:
                self.params_opt_dict[param_opt_key] = params_[param_opt_key]
            else:
                print param_opt_key,' using default value:',self.params_opt_dict[param_opt_key]
    
    def getParam(self, key_name):
        if self.params_dict.has_key(key_name):
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

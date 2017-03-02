__author__ = 'wonderland'
__version__ = '0.0'
"""
 Interface for parsing computation json params_json_file
for union testing
type: dict
{
'info':dict
'params':
	[{ 'dl_type':bool,
	   'test_proto':test_path,
	   'net_model':model_path,
	   'mean_file':mean_path},
	{ 'dl_type':bool,
	   'test_proto':test_path,
	   'net_model':model_path,
	   'mean_file':mean_path},
	...

	]
}
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
         
	self.test_params_list = list()	
	self.model_nums = 0
	self.current_model_idx = -1
        if not param_json_file is None:
            print 'loading params into memory ...'
            tic = time.time()
            params = json.load(open(params_json_file),'r')
            assert type(params) == dict, "params_json_file format %s not supported" %(type(params))
            print 'Done (t= %0.2fs)'%(time.time()-tic)
            self.parse_params(params)

    def parse_params(self,params_):
        #parsing params from params_dict
	self.test_params_list = params_['params']
	self.model_nums = len(params_['params'])
    
    def getNextParam(self):
	assert self.model_nums != 0, 'model list is empty'
	assert self.current_model_idx <= self.model_nums, 'access model out of boundary' 
	self.current_model_idx +=1
	return self.test_params_list[self.current_model_idx]
        
    def info(self):
        """
        Print information about the json file.
        :return:
        """
        for key,value in self.params_dict ['info'].items():
            print '%s: %s'%(key, value)
